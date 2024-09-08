using NumSharp;
using System.Collections.Immutable;
using System.Numerics;

namespace Neuropolator;

class ReportHistory
{
    public ImmutableList<Vector2> Positions = ImmutableList<Vector2>.Empty;
    public ImmutableList<double> Timings = ImmutableList<double>.Empty;

    public ReportHistory() { }
    public ReportHistory(Vector2 position, double time)
    {
        Positions = ImmutableList.Create(position);
        Timings = ImmutableList.Create(time);
    }

    public ReportHistory AddOneAndTrim(Vector2 position, double now, double cutoff = 10.0)
    {
        var cutoffTime = now - cutoff;
        var cutoffIndex = Timings.FindIndex(t => t > cutoffTime);
        if (cutoffIndex == -1)
            return new ReportHistory(position, now);
        else
        {
            var range = Range.StartAt(cutoffIndex);
            return new ReportHistory
            {
                Positions = Positions.Take(range).ToImmutableList().Add(position),
                Timings = Timings.Take(range).ToImmutableList().Add(now)
            };
        }
    }

    public ReportHistory AddPredictions(NDArray predictions, float predictionDeltaT)
    {
        var pos = Positions.Last();
        var time = Timings.Last();

        var newPositions = new List<Vector2>();
        var newTimings = new List<double>();

        for (int i = 0; i < predictions.shape[0]; i++)
        {
            pos += new Vector2(predictions[i, 0], predictions[i, 1]);
            newPositions.Add(pos);
            time += predictionDeltaT;
            newTimings.Add(time);
        }

        return new ReportHistory
        {
            Positions = Positions.AddRange(newPositions),
            Timings = Timings.AddRange(newTimings)
        };
    }

    public Vector2 SampleSingle(double time)
    {
        var histSize = Positions.Count;
        if (histSize < 2) return Vector2.Zero;

        // search from the end
        var tInd = histSize - 1;
        for (int i = histSize - 1; i >= 0; i--)
        {
            if (Timings[i] <= time)
            {
                tInd = i;
                break;
            }
        }

        // clip if out of bounds
        if (tInd <= 0) return Positions[0];
        if (tInd == histSize - 1) return Positions[tInd];

        // lerp if on the edge segments
        if (tInd == 1 | tInd == histSize - 2) return Vector2.Lerp(Positions[tInd - 1], Positions[tInd], (float)((time - Timings[tInd - 1]) / (Timings[tInd] - Timings[tInd - 1])));

        // cubic Barry-Goldman Catmull-Rom
        var knot0 = Timings[tInd - 2];
        var knot1 = Timings[tInd - 1];
        var knot2 = Timings[tInd];
        var knot3 = Timings[tInd + 1];

        var knot10 = (float)(knot1 - knot0);
        var knot21 = (float)(knot2 - knot1);
        var knot32 = (float)(knot3 - knot2);

        var knot20 = (float)(knot2 - knot0);
        var knot31 = (float)(knot3 - knot1);

        var knot0t = (float)(knot0 - time);
        var knot1t = (float)(knot1 - time);
        var knot2t = (float)(knot2 - time);
        var knot3t = (float)(knot3 - time);

        var b1 = knot2t / knot21;
        var b2 = (-1 * knot1t) / knot21;

        var a1 = b1 * knot2t / knot20;
        var a2 = b1 * (-1 * knot0t) / knot20 + b2 * knot3t / knot31;
        var a3 = b2 * (-1 * knot1t) / knot31;

        var p0 = a1 * knot1t / knot10;
        var p1 = a1 * (-1 * knot0t) / knot10 + a2 * knot2t / knot21;
        var p2 = a2 * (-1 * knot1t) / knot21 + a3 * knot3t / knot32;
        var p3 = a3 * (-1 * knot2t) / knot32;

        return p0 * Positions[tInd - 2] + p1 * Positions[tInd - 1] + p2 * Positions[tInd] + p3 * Positions[tInd + 1];
    }

    public NDArray ResamplePastDeltas(int steps, float minDeltaT, out float actualDeltaT)
    {
        NDArray sample = ResamplePast(steps + 1, minDeltaT, out actualDeltaT);
        NDArray deltas = np.zeros<float>(new int[] { steps, 2 });
        for (int i = 0; i < steps; i++)
        {
            float dX = sample[i + 1, 0] - sample[i, 0];
            float dY = sample[i + 1, 1] - sample[i, 1];
            if (float.IsFinite(dX) & float.IsFinite(dY)) // catch knot overlaps
            {
                deltas[i, 0] = dX;
                deltas[i, 1] = dY;
            }
        }
        return deltas;
    }

    public NDArray ResamplePast(int steps, float minDeltaT, out float actualDeltaT)
    {
        var histSize = Positions.Count;
        actualDeltaT = minDeltaT;
        if (histSize < 4) return np.zeros(new int[] { steps, 2 });

        double endTime = Timings[histSize - 1];
        double penultTime = Timings[histSize - 2];
        actualDeltaT = Math.Max(minDeltaT, (float)(endTime - penultTime));

        double startTime = endTime - actualDeltaT * steps;
        var tInterp = np.linspace(startTime, endTime, steps);

        // var tInd = np.searchsorted(Timings, tInterp);
        NDArray tInd = np.zeros<int>(new int[] { steps });
        {
            int ind = 0;
            for (int step = 0; step < steps; step++)
            {
                while ((double)Timings[ind] < (double)tInterp[step] && ind < histSize - 1) ind++;
                tInd[step] = ind;
            }
        }

        // np.take's
        NDArray knotsInterp = np.zeros<float>(new int[] { 4, steps });
        NDArray knotPositions = np.zeros<float>(new int[] { 4, steps, 2 });
        for (int step = 0; step < steps; step++)
        {
            for (int knot = 0; knot < 4; knot++)
            {
                int ind = tInd[knot] + knot - 2;
                // when clamping, same knots will cause DIV/0 in BarryGoldman fn
                ind = Math.Clamp(ind, 0, histSize - 1);
                knotsInterp[knot, step] = Timings[ind];
                knotPositions[knot, step, 0] = Positions[ind].X;
                knotPositions[knot, step, 1] = Positions[ind].Y;
            }
        }

        var knotWeights = BatchCubicBarryGoldmanWeights(knotsInterp, tInterp);
        // var weightedPositions = knotPositions * knotWeights[Slice.All, Slice.All];
        var weightedPositions = knotPositions * knotWeights[Slice.All, Slice.All, Slice.NewAxis];
        var interpPositions = np.sum(weightedPositions, axis: 0);
        return interpPositions;
    }

    public static NDArray BatchCubicBarryGoldmanWeights(NDArray knots, NDArray t)
    {
        var knots10 = (knots[1] - knots[0]).astype(np.float32);
        var knots21 = (knots[2] - knots[1]).astype(np.float32);
        var knots32 = (knots[3] - knots[2]).astype(np.float32);

        var knots20 = (knots[2] - knots[0]).astype(np.float32);
        var knots31 = (knots[3] - knots[1]).astype(np.float32);

        var knots0t = (knots[0] - t).astype(np.float32);
        var knots1t = (knots[1] - t).astype(np.float32);
        var knots2t = (knots[2] - t).astype(np.float32);
        var knots3t = (knots[3] - t).astype(np.float32);

        var b1 = knots2t / knots21;
        var b2 = (-1 * knots1t) / knots21;

        var a1 = b1 * knots2t / knots20;
        var a2 = b1 * (-1 * knots0t) / knots20 + b2 * knots3t / knots31;
        var a3 = b2 * (-1 * knots1t) / knots31;

        var p0 = a1 * knots1t / knots10;
        var p1 = a1 * (-1 * knots0t) / knots10 + a2 * knots2t / knots21;
        var p2 = a2 * (-1 * knots1t) / knots21 + a3 * knots3t / knots32;
        var p3 = a3 * (-1 * knots2t) / knots32;

        return np.stack(new NDArray[] { p0, p1, p2, p3 });
    }
}

