package local_babelnet;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.handler.AbstractHandler;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelNetQuery;
import it.uniroma1.lcl.babelnet.BabelSense;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetID;

import it.uniroma1.lcl.babelnet.BabelSynsetRelation;
import it.uniroma1.lcl.babelnet.data.BabelPointer;
import it.uniroma1.lcl.jlt.util.Language;
import it.uniroma1.lcl.kb.ResourceID;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpPrincipal;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpContext;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.OutputStream;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class Main {

	public static BabelNet bn;

	public static void main(String[] args) throws IOException {
		bn = BabelNet.getInstance();
		HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
		HttpContext context = server.createContext("/bn");
		context.setHandler(Main::handleRequest);
		server.start();
	}

	private static void handleRequest(HttpExchange exchange) throws IOException {
		URI requestURI = exchange.getRequestURI();

		// Get elements and print URI
		Map<String, String> mapping = getRequestInfo(exchange);
		String funcName = mapping.get("function");

		CompletableFuture<String> futureResult = CompletableFuture.supplyAsync(() -> {
			String lemma, synsetId;
			Language src, trg;

			switch (funcName) {
			case "get_senses":
				lemma = mapping.get("lemma");
				src = Language.valueOf(mapping.get("src"));
				trg = Language.valueOf(mapping.get("trg"));
				List<BabelSense> senses = get_senses(bn, lemma, src, trg);
				return babelSenseJson(senses);

			case "get_synset_ids":
				lemma = mapping.get("lemma");
				src = Language.valueOf(mapping.get("src"));
				trg = Language.valueOf(mapping.get("trg"));
				List<BabelSynset> synsets = get_synset_ids(bn, lemma, src, trg);
				return babelSysnsetIDsJson(synsets);

			case "get_synset":
				synsetId = mapping.get("synsetId");
				trg = Language.valueOf(mapping.get("trg"));
				List<BabelSynset> synset = get_synset(bn, synsetId, trg);
				if (!synset.isEmpty())
					return babelSynsetJson(synset.get(0));
				return "";

			case "get_outgoing_edges":
				synsetId = mapping.get("synsetId");
				List<BabelSynsetRelation> relations = get_outgoing_edges(bn, synsetId);
				return babelRelationsJson(relations);

			default:
				return "Welcome to BiVERTs Local Babelnet!";
			}
		});

		try {
			// Wait for the result to complete
			String result = futureResult.get();
			String response = result;
			exchange.sendResponseHeaders(200, response.getBytes().length);
			OutputStream os = exchange.getResponseBody();
			os.write(response.getBytes());
			os.close();
		} catch (InterruptedException | ExecutionException e) {
			// Handle any exceptions that occurred during CompletableFuture execution
			e.printStackTrace();
			// Return an error response if needed
			String response = "Error occurred: " + e.getMessage();
			exchange.sendResponseHeaders(500, response.getBytes().length);
			OutputStream os = exchange.getResponseBody();
			os.write(response.getBytes());
			os.close();
		}
	}

	private static String babelRelationsJson(List<BabelSynsetRelation> relations) {
		String result = "[";
		String strRelation = "";
		for (BabelSynsetRelation relation : relations) {
			strRelation = "{\"language\":\"" + relation.getLanguage() + "\", \"pointer\":{\"shortName\":\""
					+ relation.getPointer().getShortName().replace("\"", "'") + "\", \"relationGroup\":\""
					+ relation.getPointer().getRelationGroup() + "\"}, \"target\":\"" + relation.getTarget() + "\"},";
			result += strRelation;
		}
		
		if(result.length()>1)
			result = result.substring(0, result.length() - 1) + "]";
		else
			result = "[]";
		return result;
	}

	private static String babelSynsetJson(BabelSynset synset) {
		List<BabelSense> senses = synset.getSenses();
		String result = babelSenseJson(senses);

		return "{\"senses\":" + result + "}";
	}

	private static String babelSysnsetIDsJson(List<BabelSynset> synsets) {
		String result = "[";
		String strSynset = "";
		for (BabelSynset babelSynset : synsets) {
			strSynset = "{\"id\":\"" + babelSynset.getID() + "\", \"pos\":\"" + babelSynset.getPOS() + "\", \"source\":\""
					+ babelSynset.getSenseSources() + "\"},";
			result += strSynset;
		}
		
		if(result.length()>1)
			result = result.substring(0, result.length() - 1) + "]";
		else
			result = "[]";
		return result;
	}

	private static String babelSenseJson(List<BabelSense> senses) {
		String result = "[";
		String strSense = "";
		
		for (BabelSense sense : senses) {
			strSense = "{\"type\":\"BabelSense\", \"properties\":{\"fullLemma\":\"" + sense.getFullLemma().replace("\"", "'") + "\", \"source\":\""
					+ sense.getSource() + "\", \"senseKey\":\"" + sense.getSensekey() + "\", \"language\":\"" + sense.getLanguage()
					+ "\", \"pos\":\"" + sense.getPOS() + "\", \"idSense\":\"" + sense.getID() + "\"}}";
			result += strSense + ",";
		}
		
		if(result.length()>1)
			result = result.substring(0, result.length() - 1) + "]";
		else
			result = "[]";
		return result;
	}

	private static Map<String, String> getRequestInfo(HttpExchange exchange) {
		System.out.println("-- headers --");
		Headers requestHeaders = exchange.getRequestHeaders();
		requestHeaders.entrySet().forEach(System.out::println);

		System.out.println("-- principle --");
		HttpPrincipal principal = exchange.getPrincipal();
		System.out.println(principal);

		System.out.println("-- HTTP method --");
		String requestMethod = exchange.getRequestMethod();
		System.out.println(requestMethod);

		System.out.println("-- query --");
		URI requestURI = exchange.getRequestURI();
		String query = requestURI.getQuery();
		System.out.println(query);

		Map<String, String> mapping = getQueryMap(query);
		System.out.println(mapping);

		return mapping;
	}

	public static Map<String, String> getQueryMap(String query) {
		String[] params = query.split("&");
		Map<String, String> mapping = new HashMap<String, String>();

		for (String param : params) {
			String name = param.split("=")[0];
			String value = param.split("=")[1];
			mapping.put(name, value);
		}
		return mapping;
	}

	public static List<BabelSense> get_senses(BabelNet bn, String word, Language src, Language trg) {
		BabelNetQuery query = new BabelNetQuery.Builder(word).from(src).to(trg).build();
		List<BabelSense> senses = bn.getSensesFrom(query);
		return senses;
	}

	public static List<BabelSynset> get_synset_ids(BabelNet bn, String word, Language src, Language trg) {
		BabelNetQuery query = new BabelNetQuery.Builder(word).from(src).to(trg).build();
		List<BabelSynset> synsets = bn.getSynsets(query); 
		return synsets;
	}

	public static List<BabelSynset> get_synset(BabelNet bn, String synsetId, Language trg) {
		BabelNetQuery query = new BabelNetQuery.Builder(new BabelSynsetID(synsetId)).to(trg).build();
		List<BabelSynset> synset = bn.getSynsets(query);

		return synset;
	}

	public static List<BabelSynsetRelation> get_outgoing_edges(BabelNet bn, String synsetId) {
		BabelNetQuery query = new BabelNetQuery.Builder(new BabelSynsetID(synsetId)).build();
		List<BabelSynset> synset = bn.getSynsets(query);
		if (!synset.isEmpty()) {
			BabelSynset s = synset.get(0);
			List<BabelSynsetRelation> relations = s.getOutgoingEdges(BabelPointer.ANY_HYPERNYM);
			return relations;
		}
		
		return new ArrayList<BabelSynsetRelation>();
	}

}
