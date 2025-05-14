using System.Collections.Generic;

namespace core.data
{
    public class FeaturesData
    {
        public FeaturesData() { }

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
