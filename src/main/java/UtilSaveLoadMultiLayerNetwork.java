import org.apache.log4j.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;

public class UtilSaveLoadMultiLayerNetwork {
    final static Logger logger = Logger.getLogger(UtilSaveLoadMultiLayerNetwork.class);


    public void save(MultiLayerNetwork net, String filename) throws Exception {
        net.init();
        File locationToSave = new File(filename);
        boolean saveUpdater = true;
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);
    }

    public MultiLayerNetwork load(String filename) throws Exception {
        File locationToSave = new File(filename);
        return ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    }

}