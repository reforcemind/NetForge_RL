"""Host / CVE / token codebooks shared by the Python gym and JAX kernels."""

N_HOSTS = 100
STATUS_CODES = ('online', 'isolated', 'kernel_panic')
PRIVILEGE_CODES = ('None', 'User', 'Root')
DECOY_CODES = ('inactive', 'active', 'Apache', 'SSHD', 'Tomcat')
INTEGRITY_CODES = ('clean', 'compromised', 'kinetic_destruction')
CVE_CODES = (
    'MS17-010',
    'CVE-2019-0708',
    'CVE-2021-44228',
    'V4L2',
    'CVE-2010-2772',
    'Stuxnet_0day',
)
N_CVE = len(CVE_CODES)
TOKEN_CODES = ('Enterprise_Admin_Token', 'Local_Admin_DMZ', 'Local_Admin_Corporate')
N_TOKEN = len(TOKEN_CODES)
OS_OTHER, OS_WINDOWS, OS_LINUX, OS_PLC = (0, 1, 2, 3)


def os_family_code(os_str) -> int:
    text = str(os_str or '')
    if 'Windows' in text:
        return OS_WINDOWS
    if 'Linux' in text:
        return OS_LINUX
    if 'PLC' in text:
        return OS_PLC
    return OS_OTHER


def encode(value, codebook) -> int:
    try:
        return codebook.index(value)
    except ValueError:
        return 0


def decode(code, codebook):
    idx = int(code)
    if 0 <= idx < len(codebook):
        return codebook[idx]
    return codebook[0]


# Private aliases used by the conversion layer.
_os_family_code = os_family_code
_encode = encode
_decode = decode
