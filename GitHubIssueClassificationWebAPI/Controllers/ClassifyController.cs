using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using GitHubIssueClassification;

namespace GitHubIssueClassificationWebAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ClassifyController : ControllerBase
    {
        private readonly PredictionEnginePool<GitHubIssue, IssuePrediction> _predictionEnginePool;

        public ClassifyController(PredictionEnginePool<GitHubIssue, IssuePrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [HttpPost]
        public ActionResult<string> Post([FromBody] GitHubIssue input)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest();
            }

            IssuePrediction prediction = _predictionEnginePool.Predict(modelName: "GitHubIssueClassificationModel", example: input);
            string issue = prediction.Area;
            return Ok(issue);
        }
    }
}
