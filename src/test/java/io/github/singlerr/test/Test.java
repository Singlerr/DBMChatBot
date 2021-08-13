package io.github.singlerr.test;
import io.github.singlerr.chat.dl4j.NetworkBuilder;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

/**GravesLSTM 문자 모델링 예제
 * @author Alex Black
예제: 한 번에 한 문자 씩 텍스트를 생성하려면 LSTM RNN으로 학습해야한다.
이 예제는 Andrej Karpathy의 블로그 게시물을 참고했다.
"The Unreasonable Effectiveness of Recurrent Neural Networks"
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
이 예제는 Project Gutenberg에서 다운로드 한 Complete Works of William Shakespeare를 사용해서 교육하는 내용이다.
해당 예제를 이용해서 다른 텍스트에 대해서 비교적 쉽게 구현할 수 있을 것이다.
DL4J로 RNN을 구현하는 내용은 아래 링크들을 참고하자.
http://deeplearning4j.org/usingrnns
http://deeplearning4j.org/lstm
http://deeplearning4j.org/recurrentnetwork
 */

public class Test {
    public static void main( String[] args ) throws Exception {
        NetworkBuilder builder = NetworkBuilder.getInstance();
        builder.initialize(new File("data.csv"));
    }

    /** 셰익스피어 학습 데이터를 다운로드하여 로컬에 저장한다 (임시 디렉토리). 그런 다음 텍스트를 기반으로 벡터화를 수행하는 간단한 DataSetIterator를 설정하고 반환한다.
     * @param miniBatchSize 각 학습 미니 배치의 텍스트 세그먼트 수
     * @param sequenceLength 각 텍스트 세그먼트별로 문자 길이
     */
    public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception{
        //학습데이터 원본 이름 : The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 백만 자
        //https://www.gutenberg.org/ebooks/100
        String url = "https://raw.githubusercontent.com/Singlerr/DeepLearning/main/pg.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/isang.txt";	//Storage location from downloaded file
        File f = new File(fileLocation);
        if( !f.exists() ){
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//다운로드에 문제가 생기면 에러를 던진다

        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//허용되는 문자는 무엇입니까? 기타는 제거된다.
        return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }

    /** 네트워크로부터의 샘플을 생성한다 (null의 경우는, 옵션). 초기화를 사용하여 확장 / 계속하려는 시퀀스로 RNN을 '초기화'할 수 있다.<br>
     *  초기화는 모든 샘플에 사용된다.
     * @param initialization String형, null 일 가능성이있다. null의 경우, 모든 샘플의 초기화로서 무작위의 문자를 선택한다.
     * @param charactersToSample 신경망에서 샘플링 할 문자 수 (초기화 제외)
     * @param net 하나 이상의 GravesLSTM / RNN 레이어와 softmax 출력 레이어가있는 MultiLayerNetwork
     * @param iter CharacterIterator. 인덱스에서 문자로 이동하는 데 사용된다.
     */
    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
        //초기화를 설정하자. 초기화가없는 경우 : 임의 문자 사용
        if( initialization == null ){
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //초기화를 위한 입력 데이터 생성
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for( int i=0; i<init.length; i++ ){
            int idx = iter.convertCharacterToIndex(init[i]);
            for( int j=0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

        //한 번에 한 문자 씩 신경망에서 샘플링 (및 피드 샘플을 입력으로 다시 입력) (모든 샘플에 적용 가능)
        //샘플링은 여기에서 병렬로 수행된다.
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	//마지막 단계의 출력을 가져온다.

        for( int i=0; i<charactersToSample; i++ ){
            //이전 출력에서 샘플링하여 다음 입력 설정
            INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
            //출력은 확률 분포이다. 생성하려는 각 예제에 대한 샘플을 새 입력에 추가해주자.
            for( int s=0; s<numSamples; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//다음단계 입력 준비
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//샘플 된 문자를 StringBuilder에 추가 (사람이 읽을 수있는 출력).
            }

            output = net.rnnTimeStep(nextInput);	//정방향 전달의 한단계 수행
        }

        String[] out = new String[numSamples];
        for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
        return out;
    }

    /** 불연속 클래스에 대한 확률 분포가 주어지면, 분포로부터 샘플링하고 생성 된 클래스 인덱스를 반환한다.
     * @param distribution 클래스에 대한 확률 분포. 1.0에 합계되어야 함
     */
    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = rng.nextDouble();
        double sum = 0.0;
        for( int i=0; i<distribution.length; i++ ){
            sum += distribution[i];
            if( d <= sum ) return i;
        }
        //분포가 유효한 확률 분포라면 결코 일어나지 않아야한다.
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}