from cx_Freeze import setup, Executable

# Define the script/module to be packaged
target = Executable(
    script="main.py",  # Your main program script file
    base="Win32GUI",  # Use "Win32GUI" as the base if your program is a GUI application
    icon="logo.ico"  # Optional: Your program icon file
)

# Set the build options
options = {
    "build_exe": {
        "packages": [],  # Additional packages to include
        "include_files": [
            "images",
            "utils",
            "models",
            "weights",
            "Detection Platform.ui"
        ]  # Additional files to include (e.g., icons, data files)
    }
}

# Perform the build
setup(
    name="RS Detection Platform",  # Your application name
    version="1.2",  # Version number
    description="For Roller Shutter Detection in THL",  # Description
    executables=[target],  # Executable(s) to build
    options=options
)