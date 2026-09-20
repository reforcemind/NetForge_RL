import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = 'NetForge'
author = 'Igor Jankowski'
copyright = '2026, Igor Jankowski'
release = '4.0.0'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
]

myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath']
myst_heading_anchors = 2
myst_fence_as_directive = ['mermaid']
myst_suppress_warnings = ['myst.header']
source_suffix = {'.md': 'markdown', '.rst': 'restructuredtext'}

html_theme = 'furo'
html_title = 'NetForge'
html_logo = '_static/figures/mark.svg'
html_static_path = ['_static']
templates_path = ['_templates']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
html_copy_source = False
html_show_sphinx = False

html_theme_options = {
    'navigation_with_keys': True,
    'source_repository': 'https://github.com/reforcemind/NetForge_RL/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/reforcemind/NetForge_RL',
            'html': """
<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
""",
            'class': '',
        },
    ],
    'light_css_variables': {
        'color-brand-primary': '#7b2733',
        'color-brand-content': '#7b2733',
        'color-background-primary': '#f7efdc',
        'color-background-secondary': '#f1e6cc',
        'color-background-hover': '#ece0c4',
        'color-background-border': '#e2d4b6',
        'color-foreground-primary': '#2b2521',
        'color-foreground-secondary': '#5c5248',
        'color-foreground-muted': '#8a7d6d',
        'color-code-background': '#f2e8d0',
        'color-code-foreground': '#5e1f2b',
        'color-admonition-background': '#f1e6cc',
        'color-highlight-on-target': '#fcefd4',
        'font-stack': "'Libre Franklin', ui-sans-serif, system-ui, sans-serif",
        'font-stack--monospace': "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
        'font-stack--headings': "'Newsreader', Georgia, 'Times New Roman', serif",
        'sidebar-caption-font-size': '0.72rem',
    },
    'dark_css_variables': {
        'color-brand-primary': '#dd94a0',
        'color-brand-content': '#e3a6b0',
        'color-background-primary': '#211b19',
        'color-background-secondary': '#2a2320',
        'color-background-hover': '#342a25',
        'color-background-border': '#3b302b',
        'color-foreground-primary': '#ece0cb',
        'color-foreground-secondary': '#c9bba6',
        'color-foreground-muted': '#9a8b78',
        'color-code-background': '#2a2320',
        'color-code-foreground': '#e6c9b8',
        'color-admonition-background': '#2a2320',
        'font-stack': "'Libre Franklin', ui-sans-serif, system-ui, sans-serif",
        'font-stack--monospace': "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
        'font-stack--headings': "'Newsreader', Georgia, 'Times New Roman', serif",
    },
}

exclude_patterns = [
    '_build',
    'README.md',
    'stylesheets',
]

autodoc_mock_imports = [
    'jax',
    'jaxlib',
    'ray',
    'torch',
    'transformers',
    'sentence_transformers',
    'moviepy',
    'docker',
    'scapy',
    'minari',
    'h5py',
    'matplotlib',
    'stable_baselines3',
]
napoleon_google_docstring = True
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
}

nitpicky = False
suppress_warnings = ['myst.header']
