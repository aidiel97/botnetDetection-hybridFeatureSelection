import subprocess

libraries = [
    'pandas',              # For data manipulation
    'python-dotenv',       # For managing environment variables (optional)
    'scikit-learn',        # For machine learning algorithms and metrics
    'matplotlib',          # For data visualization
    'numpy',               # For numerical operations
    'scipy',               # For statistical calculations (e.g., Kendall Tau)
]

for library in libraries:
    try:
        subprocess.check_call(['pip', 'install', library])
        print(f'Successfully installed {library}.')
    except subprocess.CalledProcessError as e:
        print(f'Error installing {library}: {e}')
