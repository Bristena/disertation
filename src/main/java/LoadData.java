import model.Data;
import model.DogParts;
import org.apache.log4j.Logger;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.Iterator;
import java.util.List;

public class LoadData {
    final static Logger logger = Logger.getLogger(LoadData.class);


    int PART_FLIP_IDXS[][] = {{0, 1}, {3, 7}, {4, 6}};
    private int image_size = 128;
    private int num_channels = 3;

    public Data loadData(List<String> imageList) {
        logger.warn("loading image data");
        Data data = new Data();
        INDArray x = Nd4j.zeros(imageList.size(), num_channels, image_size, image_size);
        INDArray y = Nd4j.zeros(imageList.size(), 16);

        for (int ind = 0; ind < imageList.size(); ind++) {
            Mat img = Highgui
                    .imread(imageList.get(ind).replace("txt", "jpg").replace("dogParts", "dogImages"));
//                    .imread(imageList.get(ind));
            Size originalSize = img.size();
            Mat resizedImage = new Mat();
            Size sz = new Size(image_size, image_size);
            Imgproc.resize(img, resizedImage, sz);
            Core.multiply(resizedImage, new Scalar(1.0), resizedImage);
            Core.divide(resizedImage, new Scalar(255), resizedImage);
            Core.transpose(resizedImage, resizedImage);
            for (int j = 0; j < x.size(1); j++) {
                for (int k = 0; k < x.size(2); k++) {
                    for (int q = 0; q < x.size(3); q++) {
                        int[] index = {ind, j, k, q};
                        x.putScalar(index, (float) resizedImage.get(k, q)[j]); //add pixels and channel
                    }
                }
            }

            ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
//            String dogPartsPath = imageList.get(ind).replace("dogImages", "dogParts");
            DogParts dogParts = extractTrainingFaces.loadDog(imageList.get(ind).replace("jpg", "txt")); //de 8x2
//            DogParts dogParts = extractTrainingFaces.loadDog(imageList.get(ind)); //de 8x2
            INDArray pointArr = Nd4j.create(dogParts.getArray().rows(), dogParts.getArray().columns());
            INDArray indArrayFromDogParts = dogParts.getArray();
            for (int dpi = 0; dpi < indArrayFromDogParts.rows(); dpi++) {
                for (int dpj = 0; dpj < indArrayFromDogParts.columns(); dpj++) {
                    int[] dpindex = {dpi, dpj};
                    pointArr.putScalar(dpindex, indArrayFromDogParts.getFloat(dpi, dpj));
                }
            }
            double x_scale = image_size * 1.0 / originalSize.width;
            double y_scale = image_size * 1.0 / originalSize.height;
            for (int row = 0; row < pointArr.rows(); row++) {
                int indexPointAttr[] = {row, 0};
                double val = ((pointArr.getDouble(row, 0) * x_scale) - (image_size / 2)) / (image_size / 2);
                pointArr.putScalar(indexPointAttr, val);
            }
            for (int row = 0; row < pointArr.rows(); row++) {
                int indexPointAttr[] = {row, 1};
                double val = ((pointArr.getDouble(row, 1) * y_scale) - (image_size / 2)) / (image_size / 2);
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


    public MultiLayerNetwork trainNetwork(Data data) {
        final int numInputs = image_size * image_size * num_channels;
        int outputNum = 16;
        int hidden = 16;
        int iterations = 1000;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(hidden).build())
                .layer(1, new DenseLayer.Builder().nIn(numInputs).nOut(hidden).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(hidden).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        DataSet dataSet = new DataSet(data.getX(), data.getY());
        model.fit(dataSet);
        return model;
    }

    public MultiLayerNetwork trainConvNetwork(final Data data, String filename) throws Exception {
        int outputNum = 16;
        int iterations = 10;
        int batch = 50;
        int epochs = 600;
        logger.warn("Building model");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1).biasLearningRate(0.03)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(7, 7).activation(Activation.LEAKYRELU).nOut(16).build())
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
                .layer(15, new OutputLayer.Builder().nOut(data.getY().size(1)).build())
                .setInputType(InputType.convolutional(image_size, image_size, num_channels))
//                .cnnInputSize(image_size, image_size, num_channels)
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        filename = "wb_" + filename;
//        StoreBestModel storeBestModel = new StoreBestModel(filename);
//        UtilSaveLoadMultiLayerNetwork utilSaveLoadMultiLayerNetwork = new UtilSaveLoadMultiLayerNetwork();
//        try {
//            utilSaveLoadMultiLayerNetwork.save(model, filename);
//        } catch (Exception e) {
//            System.out.println(e);
//        }

/*        List<Pair<INDArray, INDArray>> list = new ArrayList<>();
        int numSamples = data.getX().size(0);
        for (int i = 0; i < numSamples / batch; i++) {
            INDArray currentX = data.getX().get(NDArrayIndex.interval(i * batch, (i + 1) * batch), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray currentY = data.getY().get(NDArrayIndex.interval(i * batch, (i + 1) * batch), NDArrayIndex.all());
            list.add(new Pair<INDArray, INDArray>(currentX, currentY));
        }

        if (numSamples % batch != 0) {
            INDArray currentX = data.getX().get(NDArrayIndex.interval((numSamples / batch) * batch, numSamples), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray currentY = data.getY().get(NDArrayIndex.interval((numSamples / batch) * batch, numSamples), NDArrayIndex.all());
            list.add(new Pair<INDArray, INDArray>(currentX, currentY));
        }*/

        DataSet dataSet = new DataSet(data.getX(), data.getY());
        List<DataSet> dataSets = dataSet.batchBy(batch);
        Iterator<DataSet> iterator = dataSets.iterator();

        model.init();
        logger.warn("Train model");

        model.setListeners(new ScoreIterationListener(iterations));
        UtilSaveLoadMultiLayerNetwork uslmln = new UtilSaveLoadMultiLayerNetwork();
        for (int i = 0; i < epochs; i++) {
            logger.warn("Started epoch " + i);
//            //Initialize the user interface backend
//            UIServer uiServer = UIServer.getInstance();
//
//            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
//
//            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//            uiServer.attach(statsStorage);
//
//            //Then add the StatsListener to collect this information from the network, as it trains
//            model.setListeners(new StatsListener(statsStorage));
            model.fit(iterator.next());
            logger.warn("*** Completed epoch" + i + " ***");

            logger.warn("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while (iterator.hasNext()) {
                DataSet ds = iterator.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
                uslmln.save(model, "progres.bin");
            }
            logger.warn(eval.stats());
            iterator = dataSets.iterator();
            System.gc();
        }
        return model;
    }
}
