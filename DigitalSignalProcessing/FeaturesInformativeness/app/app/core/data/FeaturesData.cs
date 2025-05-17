using System.Collections.Generic;

namespace app.core.data
{
    public class FeaturesData
    {
        public FeaturesData() { }

        public List<string> nameList { get; set; }
        public List<Image> imageList { get; set; }

        public void addImage(Image image)
        {
            if (imageList == null)
                imageList = new List<Image>();

            imageList.Add(image);
        }
    }

    public class Image
    {
        public int classIndex { get; set; }
        public List<double> featureList { get; set; }
    }
}
