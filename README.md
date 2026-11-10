# sun-bake

A tool to convert *those* models into editable PSD files, and back into models.

### Getting Started

Download and install Python 3.9 or higher.

1. Download this entire repository as a ZIP file.
2. Go to this site and download the official Cubism SDK for Native. You can use a temp email.
3. Extract `Live2DCubismCore.dll` from `Core\dll\windows\x86_64\`, placing it in the same directory as main.py.
4. Run `run.bat`. This will tell you if you are missing anything.

There are some instructions within the GUI of this tool on what to do. Generally, you do not need to change any default settings.

Please note, if you intend on doing a reverse conversion, start the conversion with `Write .psd.meta.json` enabled, and keep the generated file next to the PSD when converting back.

### Disclaimers

While this tool might fall within redistributable allowance, I have deliberately chosen NOT to include the SDK binaries. You are to source these yourself (it's pretty damn easy to do it too).

This tool has been tested with the models of a certain catgirl gacha game. It is not guaranteed to work with other models.