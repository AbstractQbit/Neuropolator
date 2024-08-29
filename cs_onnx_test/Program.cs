using System.Diagnostics;
using System.Numerics;
// using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace cs_onnx_test;

// class Program
// {
//     static void Main(string[] args)
//     {
//         Console.WriteLine("Hello, World!");

//         var options = new SessionOptions
//         {
//             InterOpNumThreads = 1,
//             IntraOpNumThreads = 1,
//             GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
//         };

//         using var session = new InferenceSession("../notebooks/aaaaa.onnx", options);
//         // using var session = new InferenceSession("../notebooks/dilated3_ar5steps.onnx", options);


//         var data = new float[504];
//         // var shape = new long[3] { 1, 2, 252 };
//         var shape = new long[3] { 1, 2, 249 };

//         // data[250] = -0.05f;
//         // data[502] = 0.05f;
//         // data[251] = -0.1f;
//         // data[503] = 0.1f;
//         data[247] = 0.5f;
//         data[248] = 0.3f;
//         data[495] = -0.1f;
//         data[496] = -0.5f;
//         data[497] = -0.6f;

//         using var ortInput = OrtValue.CreateTensorValueFromMemory(data, shape);

//         using var runOptions = new RunOptions();

//         var inputName = session.InputMetadata.Keys.First();

//         var inputs = new Dictionary<string, OrtValue> {
//             { inputName,  ortInput }
//         };

//         var outputName = session.OutputMetadata.Keys.First();

//         // Run inference
//         using var results = session.Run(runOptions, inputs, new List<string> { outputName });

//         var result = results.First()!.GetTensorDataAsSpan<float>();//.AsTensor<float>();
//         // Vector2 vec_result = new Vector2(result[0, 0, 0], result[0, 1, 0]);

//         Console.WriteLine("Result:");
//         // foreach (var item in result)
//         // {
//         //     Console.WriteLine(item);
//         // }
//         for (int i = 0; i < 5; i++)
//         {
//             Console.Write("{0}, ", result[i]);
//         }
//         Console.WriteLine();
//         for (int i = 0; i < 5; i++)
//         {
//             Console.Write("{0}, ", result[i+5]);
//         }
//         Console.WriteLine();

//         for (int i = 0; i < 10; i++)
//         {

//             var stopwatch = new Stopwatch();
//             stopwatch.Start();

//             using var _ = session.Run(runOptions, inputs, new List<string> { outputName });

//             stopwatch.Stop();

//             Console.WriteLine("Time: {0}ms", stopwatch.Elapsed.TotalSeconds * 1000);
//         }



//     }
// }
