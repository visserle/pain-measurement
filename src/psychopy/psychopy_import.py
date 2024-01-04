import logging
import importlib
import subprocess
import sys

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

def psychopy_import(package_name: str):
    """
    Helper function to import/install a package from the psychopy builder with the default psychopy environment.

    If the specified package is not found in the psychopy env, the script attempts to install it using pip.

    Note: If you are running this function for the first time in a psychopy script, it will most likely throw an error. Simply run the script again and it should work.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        try:
            # Try to install the package using pip in a subprocess
            process = subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            if process.returncode != 0:
                raise Exception("pip installation failed")
            logger.info(f"Successfully installed '{package_name} using pip.")
            return importlib.import_module(package_name)
        except Exception as exc:
            logger.error(f"Failed to install and import '{package_name}': {exc}")
            raise
