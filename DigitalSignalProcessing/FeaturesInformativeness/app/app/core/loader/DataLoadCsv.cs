using app.core.data;
using DynamicData;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Text;
using app.core.utils;
using System;

namespace app.core.loader
{
    public class DataLoadCsv : IDataLoader
    {
        protected string delimeter = ";";

        public bool isValidLoader(string fileName)
        {
            return fileName != null && fileName.EndsWith(".csv");
        }

        public FeaturesData LoadData(string url)
        {
            var data = new FeaturesData();

            string[] lines = File.ReadAllLines(url, Encoding.Default);

            bool isFirstStringFeatureNameHeader = FeautureInformativenessUtils.isStringContainsFeautureNames(lines[0]);
            int dataStartInd = 0;
            if (isFirstStringFeatureNameHeader)
                dataStartInd = 1;

            int imageLength = 0;
            for (int i = dataStartInd; i < lines.Length; i++)
            {
                string[] parts = lines[i].Split(delimeter).Select(x => x.Replace(".", ",")).Where(x => x != "").ToArray();
                if (parts.Length == 0)
                    continue;

                imageLength = parts.Length;
                int classLabel = Convert.ToInt32(Math.Floor(double.Parse(parts[0])));
                var features = parts.Skip(1).Select(double.Parse).ToList();
                data.addImage(new Image { classIndex = classLabel, featureList = features });
            }

            if (isFirstStringFeatureNameHeader)
                data.nameList = normalizeFeatureNames(lines[0].Split(delimeter));
            else
                data.nameList = normalizeFeatureNames(new string[imageLength]);


            return data;
        }

        protected List<string> normalizeFeatureNames(string[] featuresNamesFromFile)
        {
            List<string> featureNames = new List<string>(featuresNamesFromFile.Length);
            for (int i = 0; i < featuresNamesFromFile.Length; i++)
            {
                if (FeautureInformativenessUtils.isStringContainsFeautureNames(featuresNamesFromFile[i]))
                    featureNames.Add(featuresNamesFromFile[i]);
                else
                    featureNames.Add("Feature #" + i);
            }

            return featureNames;
        }
    }
}