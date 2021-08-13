package io.github.singlerr.chat.dl4j;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ICSVSentenceIterator implements SentenceIterator {
    private SentencePreProcessor sentencePreProcessor;
    private final BufferedReader reader;
    private List<String> lines;
    private Iterator<String> iter;
    public ICSVSentenceIterator(File input) throws IOException, InterruptedException {
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(input),"euc-kr"));
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
        return iter.next();
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

}
