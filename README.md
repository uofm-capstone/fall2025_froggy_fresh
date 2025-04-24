# README

## Team Leapfrog Spring Semester Documents

* Timesheet:
<https://docs.google.com/spreadsheets/d/1kc0mcMinUaesgds4kaM9AclEbhH363xrw8v5fDTNY8s/edit?usp=sharing>

* Team Contract:
<https://livememphis-my.sharepoint.com/:w:/g/personal/nmosman_memphis_edu/EZlw_2KXH71DrtBlCKx9Gr8B8JAPpD8BHgSKlb1FQRQiTg?e=7ufjr1>

* Client Meeting Notes:
<https://livememphis-my.sharepoint.com/:w:/r/personal/nmosman_memphis_edu/Documents/4-9-2025%20Meeting%20Notes.docx?d=w5a93dc5fa2504613942f078a000cbe7a&csf=1&web=1&e=aUBctN>

* Sprint 3 Demo Day ppt:
 <https://docs.google.com/presentation/d/1WJAErGWY4A57GtHuTaIDCBTBjq8lHj1J-tyWdWbYcH4/edit?usp=sharing>
  
## Frog Capture AI Recognizer

This project aims to develop an automated deep learning system that detects frogs in image captured from frog traps.
By streamlining data collection and analysis (along with more possibilities), the tool will help researchers at Memphis Zoo monitor frog behavior and populations more efficiently.

Results are stored in your Documents folder.

## Instructions

### Running development version

1. Check if npm is installed with `npm --version` and if it is not installed, install it from package manager or the website
2. Download and install the [UV package manager](https://docs.astral.sh/uv/) from package manager or their website. Make sure UV is in your Path.
3. Clone the repository from the github
4. cd into the repository
5. run `npm install` to install javascript dependencies
6. run `uv sync --project ./backend` to create a `./backend/.venv` folder and install python dependencies
7. run `npm run build` to compile `./src/main/main.ts` into `./electron/main.js` (also build the static vite `.tsx` for packaging)
8. to launch the application run `npm run dev`

### Building Electron application to binary

You can only compile an application for the operating system you're currently using! This means if you want to compile for Windows you need to set this up on Windows (and not WSL).

1. Make sure you've followed the development version instructions so you've `npm install`ed and `uv sync --project ./backend`ed.
2. Run `npm run build` to generate `main.ts` and static Vite files in a `./dist` folder.
3. Run `npm run package` to try to package the application into `./out/leapfrog-OPERATING_SYSTEM/leapfrog-app` (`.exe` on windows)
