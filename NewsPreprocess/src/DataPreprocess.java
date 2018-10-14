import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.io.*;
/**
 * @author Qiannan Cheng
 * 
 * News Data Preprocessing (make use of Lucene):
 * 1. tokenization
 * 2. stemming: poter stemmer
 *    normalization: lowercase
 * 3. remove stopwords
 * 4. remove punctuations
 * 5. remove digital
 */
public class DataPreprocess {
	
	/**
	 * Traversing and Save
	 * @param path
	 * @return
	 * @throws InterruptedException
	 * @throws IOException
	 */
	public static String batch_process(String path) throws InterruptedException, IOException {
		
		String savedir="../Data/preprocessed_news/";
		String doc_path = null;
		File datafile = new File(path);
		//traversing all class_folders in a data file
		File[] files = datafile.listFiles();
		for(File file: files){
			String floder_name=file.getName();
			//create a new class_floder in savedir
			File floder = new File(savedir+floder_name);
			floder.mkdir();
			//traversing all news in a class_floder
			File[] docs = file.listFiles();
			for(File doc: docs){
				String doc_name=doc.getName();
				doc_path=doc.getPath();
				System.out.println(doc_path);
				String content = pre_process(doc_path);//preprocessing
				//create a new file
				File fileNew = new File(savedir+floder_name+"/"+doc_name);
				if(!fileNew.exists()) {
				      fileNew.createNewFile();
				}
				//write the preprocessed content in the new file
				BufferedWriter out = new BufferedWriter(new FileWriter(fileNew));
				out.write(content);
				out.flush();out.close();
			}
		}
		return null;	
		
	}
	
	/**
	 * Preprocess the content of a news
	 * @param doc_path
	 * @return
	 * @throws IOException
	 */
	public static String pre_process (String doc_path) throws IOException{
		String res = null;
		File doc_file = new File(doc_path);
		//get all the text content of the document
		try {
			StringBuilder sb = new StringBuilder();
			Reader reader = new InputStreamReader(new FileInputStream(doc_file));
			char[] c = new char[30];
			int length = 0;
			while ((length = reader.read(c)) != -1){ //arrived at the end of the stream
				if (length == c.length) {
					sb.append(String.valueOf(c));
				}
				else{
					int i = 0;
					while(i < length){
						sb.append(c[i]);
						i++;
					}
					
				}	
			}
			reader.close();
			res = sb.toString();
			res = res.replaceAll("\\d+"," "); //replace all digital with a space 
			res = res.replaceAll("\\s+", " "); //replace all whitespace with a single space
			res = res.trim(); //remove the blanks in the begining and the end of string  
		} catch (IOException e) {
			e.printStackTrace();
		}
		//some important preprocessing
		StringBuilder sb_reform = new StringBuilder(); //modify the string
		//tokenization, remove punctuation, lowercase, remove stopwords
		Analyzer analyzer = new StandardAnalyzer(); 
		TokenStream ts = analyzer.tokenStream("", new StringReader(res));
		ts = new PorterStemFilter(ts); //poter stemmer
		CharTermAttribute charTermAttribute = ts.addAttribute(CharTermAttribute.class);
		try {
			ts.reset();
			//move the cursor to get the next token
			//if tokenstream to the end, return flase
			while (ts.incrementToken()){ 
				sb_reform.append(charTermAttribute.toString() + " ");
				
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		ts.close();
		analyzer.close();
		return sb_reform.toString();
	
	}
	
	public static void main(String[] args) throws InterruptedException, IOException {
		// TODO Auto-generated method stub
		DataPreprocess.batch_process("../Data/20news-18828");
		System.out.println("finished!");
	}

}
