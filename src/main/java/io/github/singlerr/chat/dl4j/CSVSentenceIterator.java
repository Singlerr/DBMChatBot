package io.github.singlerr.chat.dl4j;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class CSVSentenceIterator implements SentenceIterator {
    private SentencePreProcessor sentencePreProcessor;
    private final BufferedReader reader;
    private List<String> lines;
    private Iterator<String> iter;
    private int targetIndex = 0;
    public CSVSentenceIterator(File input,int targetIndex) throws IOException, InterruptedException {
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(input),"euc-kr"));
        this.targetIndex = targetIndex;
        lines = new ArrayList<>();
        reader.readLine();//Remove first line
        String line;
        while ((line = reader.readLine()) != null)
            lines.add(line);
        iter = lines.iterator();
        reader.close();
    }

    @Override
    public String nextSentence() {
        return iter.next().split(",")[targetIndex].trim();
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public void reset() {
        iter = lines.iterator();
    }

    @Override
    public void finish() {
        iter = lines.iterator();
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return sentencePreProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor sentencePreProcessor) {
        this.sentencePreProcessor = sentencePreProcessor;
    }

    public void setTargetIndex(int targetIndex){
        this.targetIndex = targetIndex;
    }
}
