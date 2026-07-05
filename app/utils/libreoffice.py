import os
import shutil
import platform

def find_soffice() -> str:
    """
    Locates the LibreOffice executable across platforms.
    Checks (in order): env override, PATH, common install locations.
    """
    # 1. Explicit override via environment variable
    env_path = os.environ.get("SOFFICE_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Check PATH first (works if user did add it, or on most Linux/Mac setups)
    for name in ("soffice", "soffice.exe", "soffice.com"):
        found = shutil.which(name)
        if found:
            return found

    # 3. Fall back to common install locations per OS
    system = platform.system()
    candidates = []

    if system == "Windows":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            os.path.expandvars(r"%ProgramFiles%\LibreOffice\program\soffice.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\LibreOffice\program\soffice.exe"),
            os.path.expandvars(r"%LocalAppData%\Programs\LibreOffice\program\soffice.exe"),
        ]
    elif system == "Darwin":
        candidates = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/soffice",
            "/usr/lib/libreoffice/program/soffice",
            "/opt/libreoffice/program/soffice",
            "/snap/bin/libreoffice.soffice",
        ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    raise RuntimeError(
        "LibreOffice ('soffice') not found. Install it, or set the SOFFICE_PATH "
        "environment variable to the full path of soffice.exe / soffice."
    )
