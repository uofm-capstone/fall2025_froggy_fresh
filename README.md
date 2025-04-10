# README

## Team Leapfrog Spring Semester Documents

* Timesheet:
<https://docs.google.com/spreadsheets/d/1kc0mcMinUaesgds4kaM9AclEbhH363xrw8v5fDTNY8s/edit?usp=sharing>

* Team Contract:
<https://livememphis-my.sharepoint.com/:w:/g/personal/nmosman_memphis_edu/EZlw_2KXH71DrtBlCKx9Gr8B8JAPpD8BHgSKlb1FQRQiTg?e=7ufjr1>

* Client Meeting Notes:
<https://livememphis-my.sharepoint.com/:w:/g/personal/nmosman_memphis_edu/EfYpQ5fJXXZDi6GG8-I7qSUB4-0LtRjT99K3ZuacOTAlmQ?e=kwOu2e>

* Sprint 3 Demo Day ppt:
 <https://docs.google.com/presentation/d/1WJAErGWY4A57GtHuTaIDCBTBjq8lHj1J-tyWdWbYcH4/edit?usp=sharing>
  
## Frog Capture AI Recognizer

This project aims to develop an automated deep learning system that detects frogs in image captured from frog traps.
By streamlining data collection and analysis (along with more possibilities), the tool will help researchers at Memphis Zoo monitor frog behavior and populations more efficiently.

## How To Run

1. Download and install the [UV package manager](https://docs.astral.sh/uv/)
2. Set up a venv with UV `uv venv` and then you can activate it by running `./.venv/bin/activate` (you might have to give yourself permissions to run it with `chmod` or its Windows equivalent)
3. Use UV to install dependencies (read the associated documentation on how to do so, you can find the deps in `pyproject.toml`)
4. To run the Electron side, run `npm install && npm run dev`
