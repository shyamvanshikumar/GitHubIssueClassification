using System;
using System.IO;
using System.Threading.Tasks;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace BlobStorageManager
{
    class Program
    {
        static async Task Main(string[] args)
        {
            string connectionString = Environment.GetEnvironmentVariable("GitHubIssue_Connection_string");

            BlobServiceClient blobServiceClient = new BlobServiceClient(connectionString);
            string containerName = "githubissueclassificationmodel";
            BlobContainerClient containerClient = blobServiceClient.GetBlobContainerClient(containerName);

            string modelPath = "C:/Users/t-shysingh/source/repos/GitHubIssueClassification/GitHubIssueClassification/Model/model2.zip";
            string downloadPath = "D:/Example";
            string fileName = "model2.zip";

            //await DeleteContainer(containerClient);
            //await UploadFile(containerClient, fileName, modelPath);
            await DownloadFile(containerClient, fileName, downloadPath);
         

        }

        public static async Task UploadFile(BlobContainerClient containerClient, string fileName, string localFilePath)
        {
            BlobClient blobClient = containerClient.GetBlobClient(fileName);
            using FileStream uploadFileStream = File.OpenRead(localFilePath);
            await blobClient.UploadAsync(uploadFileStream, true);
            uploadFileStream.Close();
        }

        public static async Task DownloadFile(BlobContainerClient containerClient, string fileName, string downloadFilePath)
        {
            BlobClient blobClient = containerClient.GetBlobClient(fileName);
            BlobDownloadInfo download = await blobClient.DownloadAsync();

            using(FileStream downloadFileStream = File.OpenWrite(downloadFilePath))
{
                await download.Content.CopyToAsync(downloadFileStream);
                downloadFileStream.Close();
            }
        }

        public static async Task DeleteContainer(BlobContainerClient containerClient)
        {
            Console.WriteLine("Press any key to begin clean up");
            Console.ReadLine();

            await containerClient.DeleteAsync();
        }
    }
}
