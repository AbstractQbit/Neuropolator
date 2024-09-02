using System;
using System.Diagnostics;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using OpenTabletDriver.Plugin;
using OpenTabletDriver.Plugin.Attributes;
using OpenTabletDriver.Plugin.Logging;
using OpenTabletDriver.Plugin.Output;
using OpenTabletDriver.Plugin.Tablet;
using OpenTabletDriver.Plugin.Timing;

namespace Neuropolator;

// public class Neuropolator : AsyncPositionedPipelineElement<IDeviceReport>, IDisposable
// {
//     public override PipelinePosition Position => throw new NotImplementedException();


//     protected override void ConsumeState()
//     {
//         var timestamp = HPETDeltaStopwatch.RuntimeElapsed.TotalSeconds;

//         throw new NotImplementedException();
//     }

//     protected override void UpdateState()
//     {
//         throw new NotImplementedException();
//     }




//     private void test()
//     {

//     }



//     public new void Dispose()
//     {
//         Log.Debug("Neuropolator", "Disposed");
//     }

//     ~Neuropolator()
//     {
//         Dispose();
//     }
// }


class ReportHistory
{
    public double cutoff = 10.0; // seconds to keep

    public List<Vector2> positions = new();
    public List<double> timings = new();

    public void Add(Vector2 position, double now)
    {
        Prune(now);
        positions.Add(position);
        timings.Add(now);
    }

    public void Prune(double now)
    {
        var cutoff_time = now - cutoff;
        while (timings.Count > 0 && timings[0] < cutoff_time)
        {
            timings.RemoveAt(0);
            positions.RemoveAt(0);
        }
    }


    // public NDArray positions = np.zeros(new int[] { 0, 2 });
    // public NDArray timings = np.zeros(new int[] { 0 });

    // public void AddOne(Vector2 position, double now)
    // {
    //     var cutoff_time = now - cutoff;
    //     int cutoff_index = np.searchsorted(timings, cutoff_time);

    //     positions = np.hstack(positions[cutoff_index..], np.array(new float[] { position.X, position.Y }).reshape(1, 2));
    //     timings = np.hstack(timings[cutoff_index..], np.array(now));
    // }

    public NDArray ResamplePast(float maxRate, int steps)
    {
        if (timings.Count < 2) return np.zeros(new int[] { steps, 2 });

        var endTime = timings[timings.Count - 1];
        var penuntTime = timings[timings.Count - 2];
        var delta = endTime - penuntTime;
        var rate = Math.Min(maxRate, 1.0f / delta);
        var startTime = endTime - rate * steps;
        var t_interp = np.linspace(startTime, endTime, steps);

        // var t_ind = np.searchsorted(timings, t_interp, 'right').astype(int);
        // np.

        throw new NotImplementedException();
    }




}

class SplineSampler
{
    // public DenseTensor<float> Sample(double now, float rate, int steps)
    // {
    //     var result = new DenseTensor<float>(new int[] { steps, 2 }, reverseStride: false);



    // }
}