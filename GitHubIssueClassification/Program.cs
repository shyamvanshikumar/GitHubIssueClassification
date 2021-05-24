using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string trainDataPath => Path.Combine(appPath, "..", "..", "..", "Data", "issue_train.tsv");
        private static string testDataPath => Path.Combine(appPath, "..", "..", "..", "Data", "issue_test.tsv");
        private static string modelNBPath => Path.Combine(appPath, "..", "..", "..", "Model", "model.zip");
        private static string modelTwoPath => Path.Combine(appPath, "..", "..", "..", "Model", "model2.zip");


        private static MLContext mlContext;
        private static PredictionEngine<GitHubIssue, IssueScorePrediction> predEngineNB, predEngineTwo;
        private static ITransformer trainedNBModel, trainedTwoModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            mlContext = new MLContext(seed: 0);
            _trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);

            var pipeline = ProcessData();

            BuildAndTrainNBModel(_trainingDataView, pipeline);
            BuildAndTrainTwoModel(_trainingDataView, pipeline);

            //Evaluate(trainedNBModel);
            //Evaluate(trainedTwoModel,trainDataPath);
            Evaluate(trainedTwoModel, testDataPath);
            PredictIssue();

            //SaveModelAsFile(_trainingDataView.Schema, trainedNBModel, modelNBPath);
           // SaveModelAsFile(_trainingDataView.Schema, trainedTwoModel, modelTwoPath);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                           .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                           .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                           .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"));
                           //.AppendCacheCheckpoint(mlContext);
            return pipeline;
        }

        public static void BuildAndTrainNBModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingNBPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            trainedNBModel = trainingNBPipeline.Fit(trainingDataView);
        }

        public static void BuildAndTrainTwoModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingTwoPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree("Label", "Features")))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            trainedTwoModel = trainingTwoPipeline.Fit(trainingDataView);
        }

        public static void Evaluate(ITransformer trainedModel, string dataPath)
        {
            var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(dataPath, hasHeader: true);
            var transformedTestData = trainedModel.Transform(testDataView);
            
            /*IEnumerable<IssueScorePrediction> transformedDataEnumerable = mlContext.Data.CreateEnumerable<IssueScorePrediction>(transformedTestData, reuseRowObject: true);

            foreach (ScorePrediction row in transformedDataEnumerable)
            {
                foreach(float s in row.Score)
                {
                    Console.Write($"{s:0.###}   ");
                }
                Console.WriteLine(" ");
                Console.WriteLine(row.Area);
            }*/
            var testMetrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());
        }

        public static void PredictIssue()
        {
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            predEngineNB = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssueScorePrediction>(trainedNBModel);
            predEngineTwo = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssueScorePrediction>(trainedTwoModel);

            var predictionNB = predEngineNB.Predict(issue);
            var predictionTwo = predEngineTwo.Predict(issue);

            Console.WriteLine("***********************************************************************************************************");
            Console.WriteLine($"=========     Single-prediction NB model    ======    Result:{predictionNB.Area}   =======");
            foreach (float s in predictionNB.Score)
            {
                Console.Write($"{s:0.###}   ");
            }
            Console.WriteLine(" ");

            Console.WriteLine("***********************************************************************************************************");
            Console.WriteLine($"=========     Single-prediction Two model    ======    Result:{predictionTwo.Area}   =======");
            foreach (float s in predictionTwo.Score)
            {
                Console.Write($"{s:0.###}   ");
            }
            Console.WriteLine(" ");
        }

        public static void SaveModelAsFile(DataViewSchema trainingDataViewSchema, ITransformer trainedModel, string modelPath)
        {
            mlContext.Model.Save(trainedModel, trainingDataViewSchema, modelPath);
        }
    }
}
