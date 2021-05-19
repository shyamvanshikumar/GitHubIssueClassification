using Microsoft.ML;
using System;
using System.IO;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string trainDataPath => Path.Combine(appPath, "..", "..", "..", "Data", "issue_train.tsv");
        private static string testDataPath => Path.Combine(appPath, "..", "..", "..", "Data", "issue_test.tsv");
        private static string modelPath => Path.Combine(appPath, "..", "..", "..", "Model", "model.zip");


        private static MLContext mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> predEngine;
        private static ITransformer trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            mlContext = new MLContext(seed: 0);
            _trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            Evaluate();

            SaveModelAsFile(_trainingDataView.Schema);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                           .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                           .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                           .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                           .AppendCacheCheckpoint(mlContext);
            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.NaiveBayes("Label", "Features"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            trainedModel = trainingPipeline.Fit(trainingDataView);

            predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);

            /*GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = predEngine.Predict(issue);
            Console.WriteLine($"======== Single-prediction ====== Result:{prediction.Area}=======");*/

            return trainingPipeline;
        }

        public static void Evaluate()
        {
            var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
            var transformedTestData = trainedModel.Transform(testDataView);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

        }

        public static void SaveModelAsFile(DataViewSchema trainingDataViewSchema)
        {
            mlContext.Model.Save(trainedModel, trainingDataViewSchema, modelPath);
        }
    }
}
