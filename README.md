# Quality and Diversity for Real Time Strategy (RTS) games

## Overview
The aim of the project was to design an algorithm that could be used by game developers to generate a diverse set of high performing AIs for a particular RTS game.

This repository contains the code used during the design for the final PGA ME QMIX method used to generate a population of solutions for the SMAX environment from JaxMARL (https://github.com/FLAIROx/JaxMARL).

Code for other methods tried are also present and includes:
- PGA ME IQL
- PGA ME SD SAC

The file evaluation.py was used to reevaluate the solutions from a generated repertoire.

## Installation
Dependancies:
- JaxMARL (https://github.com/FLAIROx/JaxMARL)
- QDax (https://github.com/adaptive-intelligent-robotics/QDax)

To install dependancies:
```bash
pip install -r requirements.txt
```
