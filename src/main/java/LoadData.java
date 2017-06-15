import model.Data;
import model.DogParts;
import org.apache.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.opencv.core.Core.transpose;

public class LoadData {
    private final static Logger logger = Logger.getLogger(LoadData.class);

    private int image_size = 128;
    private int num_channels = 3;

    public Data loadData(List<String> imageList) {
        logger.warn("loading image data");
        Data data = new Data();
        INDArray x = Nd4j.zeros(imageList.size(), num_channels, image_size, image_size);
        INDArray y = Nd4j.zeros(imageList.size(), 16);
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();

        for (int ind = 0; ind < imageList.size(); ind++) {
            String filename = imageList.get(ind);
            Mat originalImage = Highgui
                    .imread(filename.replace("txt", "jpg").replace("dogParts", "dogImages"));
            Size originalSize = originalImage.size();
            Mat resizedImage = new Mat(image_size, image_size, num_channels);
            Imgproc.resize(originalImage, resizedImage, resizedImage.size());
//            divide(resizedImage, new Scalar(1.0 / 255.0), resizedImage);
            transpose(resizedImage, resizedImage);
            for (int j = 0; j < x.size(1); j++) {
                for (int k = 0; k < x.size(2); k++) {
                    for (int q = 0; q < x.size(3); q++) {
                        int[] index = {ind, j, k, q};
                        x.putScalar(index, (float) (resizedImage.get(k, q)[j] * 1.0 / 255.0)); //add pixels and channel
                    }
                }
            }
            DogParts dogParts = extractTrainingFaces.loadDog(imageList.get(ind).replace("jpg", "txt")); //de 8x2
            INDArray pointArr = Nd4j.create(dogParts.getArray().rows(), dogParts.getArray().columns());
            INDArray indArrayFromDogParts = dogParts.getArray();
            for (int dpi = 0; dpi < indArrayFromDogParts.rows(); dpi++) {
                for (int dpj = 0; dpj < indArrayFromDogParts.columns(); dpj++) {
                    int[] dpindex = {dpi, dpj};
                    pointArr.putScalar(dpindex, indArrayFromDogParts.getFloat(dpi, dpj));
                }
            }
            float x_scale = (float) (image_size * 1.0 / originalSize.width);
            float y_scale = (float) (image_size * 1.0 / originalSize.height);
            for (int row = 0; row < pointArr.rows(); row++) {
                int indexPointAttr[] = {row, 0};
                float val = ((pointArr.getFloat(row, 0) * x_scale) - (image_size / 2)) / (image_size / 2);
                pointArr.putScalar(indexPointAttr, val);
            }
            for (int row = 0; row < pointArr.rows(); row++) {
                int indexPointAttr[] = {row, 1};
                float val = ((pointArr.getFloat(row, 1) * y_scale) - (image_size / 2)) / (image_size / 2);
                pointArr.putScalar(indexPointAttr, val);
            }
            int path[] = {1, pointArr.shape()[0] * pointArr.shape()[1]};
            pointArr = pointArr.reshape(path);

            for (int cols = 0; cols < pointArr.columns(); cols++) {
                int[] index = {ind, cols};
                y.putScalar(index, pointArr.getFloat(0, cols));
            }
            if (ind % 500 == 0) logger.warn(ind + " IMAGES LOADED");
        }
        data.setX(x);
        data.setY(y);
        return data;
    }

    public MultiLayerNetwork trainConvNetwork(final Data data, String filename) throws Exception {
        int batch = 100;
        int iterations = data.getX().size(0) / batch + 1;
        int epochs = 600;
        logger.warn("Building model");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .iterations(iterations)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1).biasLearningRate(0.03)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(7, 7).activation(Activation.LEAKYRELU).nOut(16).build()) //rectified linear units
                .layer(1, new ConvolutionLayer.Builder(5, 5).nOut(32).activation(Activation.LEAKYRELU).build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build())
                .layer(3, new DropoutLayer.Builder(0.1).build())
                .layer(4, new ConvolutionLayer.Builder(5, 5).nOut(64).activation(Activation.LEAKYRELU).build())
                .layer(5, new ConvolutionLayer.Builder(3, 3).nOut(64).activation(Activation.LEAKYRELU).build())
                .layer(6, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build())
                .layer(7, new DropoutLayer.Builder(0.2).build())
                .layer(8, new ConvolutionLayer.Builder(3, 3).nOut(256).activation(Activation.LEAKYRELU).build())
                .layer(9, new ConvolutionLayer.Builder(3, 3).nOut(256).activation(Activation.LEAKYRELU).build())
                .layer(10, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).build())
                .layer(11, new DropoutLayer.Builder(0.2).build())
                .layer(12, new DenseLayer.Builder().nOut(1250).build())
                .layer(13, new DropoutLayer.Builder(0.75).build())
                .layer(14, new DenseLayer.Builder().nOut(1000).build())
                .layer(15, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(data.getY().size(1)).build())
                .setInputType(InputType.convolutional(image_size, image_size, num_channels))
//                .cnnInputSize(image_size, image_size, num_channels)
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        System.out.println(iterations);
        DataSet dataSet = new DataSet(data.getX(), data.getY());
        IteratorDataSetIterator iterator = new IteratorDataSetIterator(dataSet.iterator(), batch);

        model.init();
        logger.warn("Train model");

        model.setListeners(new ScoreIterationListener(iterations));
        UtilSaveLoadMultiLayerNetwork uslmln = new UtilSaveLoadMultiLayerNetwork();
        for (int i = 0; i < epochs; i++) {
            logger.warn("Started epoch " + i);
//            //Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();

            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);

            //Then add the StatsListener to collect this information from the network, as it trains
            model.setListeners(new StatsListener(statsStorage));
            iterator = new IteratorDataSetIterator(dataSet.iterator(), batch);
            model.fit(iterator);
            uslmln.save(model, filename);
            logger.warn("*** Completed epoch" + i + " ***");

/*            logger.warn("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while (iterator.hasNext()) {
                DataSet ds = iterator.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
                uslmln.save(model, filename);
            }
            logger.warn(eval.stats());*/
            System.gc();
        }
        return model;
    }
}
