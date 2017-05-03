import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Brysu on 03.05.2017.
 */
public class ReadFiles {
    /**
     * expects as input a directory
     *
     * @return a list of filenames within this directory
     */
    public List<String> getFilesInDirectory() {
        File folder = new File("your/path");
        File[] listOfFiles = folder.listFiles();
        List<String> files = new ArrayList<String>();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                files.add(listOfFiles[i].getName());
            }
        }
        return files;
    }

}
