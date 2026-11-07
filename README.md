# Seismic Detection Across the Solar System

**3rd place, NASA Space Apps Challenge 2024 (Athlone)**

## The challenge

[Seismic Detection Across the Solar System](https://www.spaceappschallenge.org/): write a program that analyses real seismic data from the Apollo missions and the Mars InSight lander, and identifies quakes within the noise.

## What we built

In 48 hours, a team of two built:

- **A neural network** that classifies segments of Apollo lunar seismic data as *quake* or *noise*.
- **A Unity 3D viewer** for exploring the detected quakes visually.

## How it works

1. The raw seismic signals are split into time windows and preprocessed.
2. A neural network trained on the catalogued quakes in the challenge dataset labels each window as quake or noise.
3. The detections are exported and loaded into the Unity viewer.

## Results and limitations

The model picked out catalogued quakes in the Apollo test data well enough to place third. It's a hackathon prototype, though, not a validated scientific tool:

- It was trained and tested on a small dataset, so it hasn't been checked on new data.
- We focused on lunar data and didn't get to apply it properly to the Mars InSight data.
- With more time, we'd compare it against classical detection methods such as STA/LTA and tune it for noisier signals.

## Team

Fabrizio De Fiore (team lead) and [teammate's name].
