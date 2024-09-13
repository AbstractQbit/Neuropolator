# Neuropolator
A plugin for [OpenTabletDriver](https://github.com/OpenTabletDriver/OpenTabletDriver) that makes use of a neural network to compensate for the lag created by interpolation.

Currently, the test filter that extrapolates positions on raw reports seems to work fine.
I planned to resample the history in the actual interpolator, but it's proven to be too expensive, so it's unusable for now.
Will rethink how to approach this soon.
