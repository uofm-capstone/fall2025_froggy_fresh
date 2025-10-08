# README

## Froggy Fresh Fall 2025 Documents
* Timesheet: https://docs.google.com/spreadsheets/d/1xcMFReWcBJ__tyjXvqYqbAp0-dRgmOaRzxluAu3lHHk
* Team Contract: https://docs.google.com/document/d/11bn6WYxDa5iVKj05U69uvbrIomSSs02lq8bIM02frT4/edit?tab=t.0
* Demo Day Slideshow: https://docs.google.com/presentation/d/1cz3R0_1_6XCYNbiV0pAiOBF-sIOWKPiGXT9J_JHUbks/edit
* Notes: https://docs.google.com/document/d/1JARno9335g6qGYY5gelcrZA_0k6v0L0IzJhCMttzEik/edit?usp=sharing

### Sprint 2
[Nicole Clark feedback for Jacob Hensley](https://github.com/user-attachments/files/22782978/Jacob.Hensley.Formal.Review.by.Nicole.Clark.pdf)
[Jacob Hensley feedback for Nicole Clark](https://github.com/user-attachments/files/22782985/Nicole.Clark.Formal.Review.by.Jacob.Hensley.pdf)


## Frog Capture AI Recognizer

This project aims to develop an automated deep learning system that detects frogs in image captured from frog traps.
By streamlining data collection and analysis (along with more possibilities), the tool will help researchers at Memphis Zoo monitor frog behavior and populations more efficiently.

Results are stored in your Documents folder.

## Instructions

### Running development version

1. Check if npm is installed with `npm --version` and if it is not installed, install it from package manager or the website
   1. On Windows installing `npm` is kinda borked rn:
      1. Go here <https://nodejs.org/en/download/> and then run the `winget` and `fnm install` lines.
      2. for some reason, `fnm install 22` doesn't add `node` to your `PATH`, so you have to add it manually.
      3. To do that, run `$env:PATH += ";C:\Users\` + YOUR USERNAME + `\AppData\Roaming\fnm\node-versions\v22.14.0\installation"` (or add that line to your `$PROFILE` so it works through shell restarts)
      4. Make new terminal session and run `node -v` to confirm you can access Node.
2. Download and install the [UV package manager](https://docs.astral.sh/uv/) from package manager or their website. Make sure UV is in your Path.
3. Clone the repository from the github
4. cd into the repository
5. run `npm install` to install javascript dependencies
6. On Linux and Mac, uncomment out the `tensorflow` line in the dependencies list in `./backend/pyproject.toml`
7. run `uv sync --project ./backend` to create a `./backend/.venv` folder and install python dependencies
8. On Windows, you need to install tensorflow manually. `cd` into `backend` and then run `uv pip install tensorflow` and then `cd` back out with `cd ..`
9. run `npm run build` to compile `./src/main/main.ts` into `./electron/main.js` (also builds the static vite `.tsx` for packaging)
10. to launch the application run `npm run dev`

### Building Electron application to binary

You can only compile an application for the operating system you're currently using! This means if you want to compile for Windows you need to set this up on Windows (and not WSL).

1. Make sure you've followed the development version instructions so you've `npm install`ed and `uv sync --project ./backend`ed.
2. Run `npm run build` to generate `main.ts` and static Vite files in a `./dist` folder.
3. Run `npm run package` to try to package the application into `./out/leapfrog-OPERATING_SYSTEM/leapfrog-app` (`.exe` on windows)
