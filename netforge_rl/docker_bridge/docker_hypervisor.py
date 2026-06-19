from __future__ import annotations

import logging
import time

from netforge_rl.docker_bridge.hypervisor_base import BaseHypervisor, HypervisorResult

logger = logging.getLogger(__name__)

_IMAGE_REGISTRY: dict[str, str] = {
    'ExploitEternalBlue': 'vulhub/samba:CVE-2017-0144',
    'ExploitBlueKeep': 'vulhub/rdp:CVE-2019-0708',
    'ExploitHTTP_RFI': 'vulhub/php:8.1-rfi',
    'ExploitRemoteService': 'ubuntu:20.04',
    'PrivilegeEscalate': 'ubuntu:20.04',
    'DumpLSASS': 'vulhub/windows-mimikatz:latest',
    'PassTheTicket': 'vulhub/windows-mimikatz:latest',
}
_FALLBACK_IMAGE = 'ubuntu:20.04'

_PAYLOAD_SCRIPTS: dict[str, str] = {
    'ExploitEternalBlue': (
        'echo "[*] Sending EternalBlue exploit packet..." && '
        'sleep 1 && '
        'echo "[+] ETERNALBLUE overwrite completed (0xC000000D)!" && '
        'echo "[*] Meterpreter session 1 opened" && '
        'echo "meterpreter > getuid" && '
        'echo "Server username: NT AUTHORITY\\\\SYSTEM"'
    ),
    'ExploitBlueKeep': (
        'echo "[*] Checking BlueKeep vulnerability..." && '
        'sleep 1 && '
        'echo "[+] Target is vulnerable." && '
        'echo "[*] Triggering kernel UAF..." && '
        'echo "[*] Meterpreter session opened."'
    ),
    'ExploitHTTP_RFI': (
        'echo "[*] Attempting RFI via GET parameter..." && '
        'sleep 0.5 && '
        'echo "[+] RFI successful. Webshell active." && '
        'echo "[*] Meterpreter session opened."'
    ),
    'ExploitRemoteService': (
        'echo "[*] Sending payload to remote service..." && '
        'sleep 0.8 && '
        'echo "[*] Command shell session opened." && '
        'echo "shell> whoami" && echo "www-data"'
    ),
    'DumpLSASS': (
        'echo "meterpreter > hashdump" && '
        'sleep 0.5 && '
        'echo "Administrator:500:aad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0:::" && '
        'echo "krbtgt:502:aad3b435b51404ee:8846f7eaee8fb117ad06bdd830b7586c:::"'
    ),
    'PassTheTicket': (
        'echo "mimikatz > kerberos::ptt ticket.kirbi" && '
        'sleep 0.3 && '
        'echo "* File: OK" && '
        'echo "[+] Ticket successfully imported. Access granted."'
    ),
}
_FALLBACK_SCRIPT = 'echo "[*] Payload dispatched." && sleep 0.5'


class DockerHypervisor(BaseHypervisor):
    """Live Docker hypervisor driver for ephemeral Vulhub containers."""

    NETWORK_NAME = 'netforge_isolated'

    def __init__(self) -> None:
        self._client = None
        self._active_containers: list = []
        self._available = self._connect()

    def dispatch(
        self,
        action_name: str,
        target_ip: str,
        target_os: str,
    ) -> HypervisorResult:

        image = _IMAGE_REGISTRY.get(action_name, _FALLBACK_IMAGE)
        script = _PAYLOAD_SCRIPTS.get(action_name, _FALLBACK_SCRIPT)

        t_start = time.perf_counter()
        container = None
        try:
            container = self._client.containers.run(
                image,
                command=f'/bin/sh -c "{script}"',
                detach=True,
                network=self.NETWORK_NAME,
                auto_remove=False,
                mem_limit='128m',
                cpu_period=100000,
                cpu_quota=25000,  # 25% of one core maximum
            )
            self._active_containers.append(container)

            result = container.wait(timeout=30)
            stdout_bytes = container.logs(stdout=True, stderr=False)
            stdout = stdout_bytes.decode('utf-8', errors='replace')
            return_code = result.get('StatusCode', 1)
            success = return_code == 0

        except Exception as exc:
            logger.error('DockerHypervisor dispatch error: %s', exc)
            stdout = f'[-] Container error: {exc}'
            return_code = 2
            success = False
        finally:
            if container is not None:
                try:
                    container.stop(timeout=5)
                    container.remove(force=True)
                    if container in self._active_containers:
                        self._active_containers.remove(container)
                except Exception:
                    pass

        latency_ms = (time.perf_counter() - t_start) * 1000

        return HypervisorResult(
            success=success,
            stdout=stdout.strip(),
            return_code=return_code,
            latency_ms=round(latency_ms, 1),
            action_name=action_name,
            target_ip=target_ip,
            target_os=target_os,
        )

    def teardown_all(self) -> None:
        """Stop and remove all active containers."""
        for container in list(self._active_containers):
            try:
                container.stop(timeout=3)
                container.remove(force=True)
            except Exception:
                pass
        self._active_containers.clear()

    def is_available(self) -> bool:
        return self._available

    def _connect(self) -> bool:
        try:
            import docker

            self._client = docker.from_env()
            self._client.ping()
            self._ensure_network()
            logger.info('DockerHypervisor: connected to Docker daemon.')
            return True
        except ImportError:
            logger.warning(
                'DockerHypervisor: `docker` SDK not installed. '
                'Run `pip install docker` to enable real-mode evaluation.'
            )
            return False
        except Exception as exc:
            logger.warning('DockerHypervisor: cannot reach daemon — %s', exc)
            return False

    def _ensure_network(self) -> None:
        """Create the isolated bridge network if it does not exist."""
        if self._client is None:
            return
        existing = [n.name for n in self._client.networks.list()]
        if self.NETWORK_NAME not in existing:
            self._client.networks.create(
                self.NETWORK_NAME,
                driver='bridge',
                internal=True,  # No external internet access — fully air-gapped
            )
            logger.info(
                'DockerHypervisor: created isolated network %s.', self.NETWORK_NAME
            )
