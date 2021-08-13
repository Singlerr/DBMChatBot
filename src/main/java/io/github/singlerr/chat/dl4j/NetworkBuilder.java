package io.github.singlerr.chat.dl4j;

import au.com.bytecode.opencsv.CSVReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.iterator.Word2VecDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.io.Files;
import reactor.core.publisher.Flux;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class NetworkBuilder {

    private static NetworkBuilder instance;

    public static NetworkBuilder getInstance() {
        if (instance == null)
            return (instance = new NetworkBuilder());
        return instance;
    }

    public void initialize(File input) throws IOException, InterruptedException {
        int inputSize = 0;
        int outputSize = 0;

        TokenizerFactory factory = new DefaultTokenizerFactory();
        factory.setTokenPreProcessor(new CommonPreprocessor());


        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(new CSVSentenceIterator(input, 0))
                .tokenizerFactory(factory)
                .build();
        vec.fit();
        File file = new File("test");
        if (!file.exists())
            file.mkdir();
        initializeLabels(input);
        Word2VecDataSetIterator iterator = new Word2VecDataSetIterator(vec, new LabelAwareFileSentenceIterator(new File(input.getParentFile(), "labels")), Arrays.asList("0", "1", "2"));
        inputSize = iterator.inputColumns();
        outputSize = iterator.totalOutcomes();
        int answerCount = 0;
        CSVRecordReader reader = new CSVRecordReader();
        reader.initialize(new FileSplit(input));
        while (reader.hasNext()) {
            reader.next();
            answerCount++;
        }
        TfidfVectorizer vectorizer;
        WordVectors vectors;
        System.out.println(String.format("%d input columns and %d output columns and %d answer count", inputSize, outputSize, answerCount));
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1337)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .nIn(inputSize).nOut(1024)
                        .encoderLayerSizes(1024, 512, 256, 128)
                        .decoderLayerSizes(128, 256, 512, 1024)
                        .lossFunction(Activation.RELU, LossFunctions.LossFunction.MSE)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .dropOut(0.8)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(1024).nOut(outputSize)
                        .activation(Activation.RELU)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(outputSize).nOut(answerCount)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .backpropType(BackpropType.Standard)
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        network.setListeners(new ScoreIterationListener(1));
        network.init();
        for (int i = 0; i < 10; i++) {
            while (iterator.hasNext()) {
                DataSet dataSet = iterator.next();
                network.fit(dataSet);
            }
            iterator.reset();
        }
    }

    private void initializeLabels(File csv) {
        File labelContainer = new File(csv.getParentFile(), "labels");
        if (!labelContainer.exists())
            labelContainer.mkdir();
        try {
            CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(csv), "euc-kr"));
            reader.readNext();
            String[] line;
            HashMap<String, List<String>> data = new HashMap<>();
            while ((line = reader.readNext()) != null) {
                String label = line[2];
                if(! data.containsKey(label)){
                    data.put(label,Arrays.asList(line[1]));
                }else{
                    List<String> list = new ArrayList<>(data.get(label));
                    list.add(line[1]);
                    data.put(label,list);
                }
            }
            data.forEach((label,lines) -> {
                File labelFolder = new File(labelContainer, label);
                if (!labelFolder.exists()) {
                    labelFolder.mkdir();
                }
                File labelFile = new File(labelFolder, "data.txt");
                if(labelFile.exists())
                    labelFile.delete();
                try {
                    BufferedWriter writer = Files.newWriter(labelFile,Charset.forName("UTF8"));
                    for(String s : lines) {
                        writer.write(s);
                        writer.newLine();
                    }
                    writer.flush();
                    writer.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private boolean isInteger(String var) {
        try {
            Integer.parseInt(var);
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }
}
