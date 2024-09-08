using OpenTabletDriver.Plugin;
using OpenTabletDriver.Plugin.Attributes;
using OpenTabletDriver.Plugin.Output;
using OpenTabletDriver.Plugin.Tablet;
using OpenTabletDriver.Plugin.Timing;
using System.Numerics;

namespace Neuropolator;

[PluginName("Neuropolator")]
public class Neuropolator : AsyncPositionedPipelineElement<IDeviceReport>, IDisposable
{
    public override PipelinePosition Position => PipelinePosition.Pixels;

    [Property("Delay offset"), Unit("ms"), DefaultPropertyValue(0.0f)]
    public float DelayOffset { get; set; }

    [Property("Prediction update time"), Unit("ms"), DefaultPropertyValue(3.0f),
     ToolTip("Duration of a lerp to the updated predictions")]
    public float PredictionUpdateTime { get; set; }

    [Property("Space scale"), DefaultPropertyValue(0.2f)]
    public float SpaceScale { get; set; }

    [Property("Reset time"), Unit("ms"), DefaultPropertyValue(50.0f),
     ToolTip("Time in milliseconds after which to reset the stroke")]
    public float ResetTime { get; set; }

    public float MinDeltaT = 1 / 200.0f;

    private static ModelRunner _runner = new("model.onnx");
    private static int _inCtx = _runner.InCtx;
    private static int[] _inShape = new[] { 2, _inCtx };

    private Vector2 _previousPosition = Vector2.Zero;
    private HPETDeltaStopwatch _reportStopwatch = new HPETDeltaStopwatch();
    private HPETDeltaStopwatch _strokeStopwatch = new HPETDeltaStopwatch();
    private ReportHistory _realHistory = new();
    private Tuple<ReportHistory, double> _currentPredictionHistory = Tuple.Create(new ReportHistory(), 0.0);
    private Tuple<ReportHistory, double> _previousPredictionHistory = Tuple.Create(new ReportHistory(), 0.0);

    protected override void ConsumeState()
    {
        if (State is IAbsolutePositionReport report)
        {
            var position = report.Position;
            // Log.Debug("Neuropolator", "Received position: " + position);

            // Reset
            if (_reportStopwatch.Restart().TotalMilliseconds > ResetTime)
            {
                _strokeStopwatch.Restart();
                _previousPosition = position;
                _realHistory = new();
            }
            var now = _strokeStopwatch.Elapsed.TotalSeconds;
            _realHistory = _realHistory.AddOneAndTrim(position, now);

            var pastDeltas = _realHistory.ResamplePastDeltas(_inCtx, MinDeltaT, out var deltaT);
            var prediction = _runner.PredictTransposed(pastDeltas);

            Tuple<ReportHistory, double> newPredictionHistory = Tuple.Create(_realHistory.AddPredictions(prediction, deltaT), now);
            _previousPredictionHistory = _currentPredictionHistory;
            _currentPredictionHistory = newPredictionHistory;
        }
        else OnEmit();
    }

    protected override void UpdateState()
    {
        if (State is IAbsolutePositionReport report && PenIsInRange())
        {
            var now = _strokeStopwatch.Elapsed.TotalSeconds;
            var (currentPredictionHistory, currentPredictionT) = _currentPredictionHistory;
            var (previousPredictionHistory, previousPredictionT) = _previousPredictionHistory;

            Vector2 position;
            if (now - currentPredictionT > PredictionUpdateTime)
            {
                position = currentPredictionHistory.SampleSingle(now + DelayOffset);
            }
            else
            {
                var alpha = (float)((now - previousPredictionT) / (currentPredictionT - previousPredictionT));
                alpha = Math.Clamp(alpha, 0.0f, 1.0f);
                var posPrev = previousPredictionHistory.SampleSingle(now + DelayOffset);
                var posCurr = currentPredictionHistory.SampleSingle(now + DelayOffset);
                position = Vector2.Lerp(posPrev, posCurr, alpha);
            }

            report.Position = position;
            State = report;
            OnEmit();
        }
    }

    public new void Dispose()
    {
        Log.Debug("Neuropolator", "Disposed");
    }
    ~Neuropolator() => Dispose();
}
