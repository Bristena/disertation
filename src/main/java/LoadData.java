import model.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class LoadData {

    private int image_size = 128;
    private int num_channels = 3;

    int PART_FLIP_IDXS[][] = {{0, 1}, {3, 7}, {4, 6}};

    public Data loadData(List<String> imageList) {
        System.out.println("loading image data");
        Data data = new Data();
        INDArray x = Nd4j.zeros(imageList.size(), num_channels, image_size, image_size);
        INDArray y = Nd4j.zeros(imageList.size(), 16);

        for (int i = 0; i < imageList.size(); i++) {
            Mat img = Highgui
                    .imread(imageList.get(i));
            Size originalSize = img.size();
            Mat resizedImage = new Mat();
            Size sz = new Size(image_size, image_size);
            Imgproc.resize(img, resizedImage, sz);
            Core.multiply(resizedImage, new Scalar(1.0), resizedImage);
            Core.divide(resizedImage, new Scalar(255), resizedImage);
            Core.transpose(resizedImage, resizedImage);
            x.getColumn(i).add((INDArray) img);
        }

//        img = imresize(img, (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)) * 1.0 / 255
//        img = img.transpose((2, 1, 0))
//        X[idx,:,:,:] = img.astype(np.float32)


        return data;
    }
}
