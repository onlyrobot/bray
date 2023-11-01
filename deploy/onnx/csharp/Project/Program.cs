﻿// See https://aka.ms/new-console-template for more information
using System.Diagnostics;
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;


var onnxModelDir = args[0];

OrtValue ConvertNpArrayToOrtValue(NDArray npArray)
{
    if (npArray.dtype == np.float32)
        return OrtValue.CreateTensorValueFromMemory(
            npArray.astype(np.float32).ToArray<float>(),
            Array.ConvertAll(npArray.shape, x => (long)x));

    if (npArray.dtype == np.float64)
        return OrtValue.CreateTensorValueFromMemory(
            npArray.astype(np.float64).ToArray<double>(),
            Array.ConvertAll(npArray.shape, x => (long)x));

    if (npArray.dtype == np.int32)
        return OrtValue.CreateTensorValueFromMemory(
            npArray.astype(np.int32).ToArray<int>(),
            Array.ConvertAll(npArray.shape, x => (long)x));

    if (npArray.dtype == np.int64)
        return OrtValue.CreateTensorValueFromMemory(
            npArray.astype(np.int64).ToArray<long>(),
            Array.ConvertAll(npArray.shape, x => (long)x));
            
    throw new Exception($"Unsupported data type {npArray.dtype}");
}

var modelPath = Path.Combine(onnxModelDir, "model.onnx");
using var session = new InferenceSession(modelPath);
Dictionary<string, OrtValue> LoadNpyFiles(string dir)
{
    // Get list of npy files in directory
    var files = Directory.GetFiles(dir, "*.npy");

    var npyFiles = new Dictionary<string, OrtValue>();

    // Load each npy file as an NDArray
    foreach (var file in files)
    {
        // Get file name without extension
        string name = Path.GetFileNameWithoutExtension(file);

        // Load npy file as NDArray
        var npArray = np.load(file);

        var ortValue = ConvertNpArrayToOrtValue(npArray);
        // Add NDArray to dictionary
        npyFiles.Add(name, ortValue);
    }

    return npyFiles;
}
var forwardInputs = LoadNpyFiles(Path.Combine(onnxModelDir, "forward_inputs"));
forwardInputs = Enumerable.Range(0, forwardInputs.Count).ToDictionary(
    x => session.InputNames[x], x => forwardInputs[x.ToString()]);
// print input information for debugging
// foreach (var input in forwardInputs)
// {
//     var tensorInfo = input.Value.GetTensorTypeAndShape();
//     Console.WriteLine($"Input {input.Key} has " +
//     $"dtype {tensorInfo.ElementDataType}, " +
//     $"shape {tensorInfo.DimensionsCount}, " +
//     $"and element count {tensorInfo.ElementCount}");
// }
var forwardOutputs = LoadNpyFiles(Path.Combine(onnxModelDir, "forward_outputs"));
// print output information for debugging
// foreach (var output in forwardOutputs)
// {
//     var tensorInfo = output.Value.GetTensorTypeAndShape();
//     Console.WriteLine($"Output {output.Key} has " +
//     $"dtype {tensorInfo.ElementDataType}, " +
//     $"shape {tensorInfo.DimensionsCount}, " +
//     $"and element count {tensorInfo.ElementCount}");
// }

using var runOptions = new RunOptions();

// Pass inputs and request the first output
// Note that the output is a disposable collection that holds OrtValues
using var outputs = session.Run(runOptions, forwardInputs, session.OutputNames);

// test session run latency
var sw = new Stopwatch();
sw.Start();
for (int i = 0; i < 100; i++)
{
    using var outputs2 = session.Run(runOptions, forwardInputs,
     session.OutputNames);
}
sw.Stop();
Console.WriteLine($"Session run latency: {sw.ElapsedMilliseconds / 100.0} ms");

if (forwardOutputs.Count != outputs.Count)
{
    Console.WriteLine($"Error: number of outputs does not match, " +
        $"origin is {forwardOutputs.Count} " +
        $"and now {outputs.Count}");
    return;
}

(float, float) CalculateError<T>(OrtValue originArray,
OrtValue targetArray) where T : unmanaged
{
    var origin = originArray.GetTensorDataAsSpan<T>();
    var target = targetArray.GetTensorDataAsSpan<T>();
    T zero = default;
    dynamic sum = zero;
    dynamic diff = zero;

    for (int j = 0; j < origin.Length; j++)
    {
        dynamic originVal = origin[j];
        dynamic targetVal = target[j];
        diff += Math.Abs(originVal - targetVal);
        sum += Math.Abs(originVal);
    }
    if (sum.Equals(zero))
        return (diff / origin.Length, 0);

    return (diff / origin.Length, (float)diff / (float)sum);
}

// calculate abs and relative error of origin and target value
Console.WriteLine($"Relative and absolute error of each output:");
for (int i = 0; i < forwardOutputs.Count; i++)
{
    var output = outputs[i];
    var originOutput = forwardOutputs[i.ToString()];

    var origin = originOutput.GetTensorTypeAndShape();
    var target = output.GetTensorTypeAndShape();
    if (origin.ElementDataType != target.ElementDataType)
    {
        Console.WriteLine($"Tensor type of {i}th output is not equal, " +
            $"origin is {origin.ElementDataType} " +
            $"and now {target.ElementDataType}");
        return;
    }
    if (origin.DimensionsCount != target.DimensionsCount)
    {
        Console.WriteLine($"Tensor shape of {i}th output is not equal, " +
            $"origin is {origin.DimensionsCount} " +
            $"and now {target.DimensionsCount}");
        return;
    }
    if (origin.ElementCount != target.ElementCount)
    {
        Console.WriteLine($"Tensor element count of {i}th output is not equal, " +
            $"origin is {origin.ElementCount} " +
            $"and now {target.ElementCount}");
        return;
    }

    float relative = 0.0f, absolute = 0.0f;
    if (origin.ElementDataType == TensorElementType.Float)
    {
        (absolute, relative) = CalculateError<float>(originOutput, output);
    }
    else if (origin.ElementDataType == TensorElementType.Double)
    {
        (absolute, relative) = CalculateError<double>(originOutput, output);
    }
    else if (origin.ElementDataType == TensorElementType.Int32)
    {
        (absolute, relative) = CalculateError<int>(originOutput, output);
    }
    else if (origin.ElementDataType == TensorElementType.Int64)
    {
        (absolute, relative) = CalculateError<long>(originOutput, output);
    }
    else
        throw new Exception($"Unsupported data type {origin.ElementDataType}");
    Console.WriteLine($"{i}th output error: {relative * 100}%, {absolute}");
}
