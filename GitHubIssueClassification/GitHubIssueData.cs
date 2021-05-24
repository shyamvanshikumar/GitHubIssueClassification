using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace GitHubIssueClassification
{
    public class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }

    public class IssueScorePrediction
    {
        [LoadColumn(0, 21)]
        [VectorType(22)]
        public float[] Score;

        [LoadColumn(7)]
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
