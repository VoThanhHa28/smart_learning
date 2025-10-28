class QueryRequest {
  final String question;
  final int topK;
  final String subject;
  final String courseId;

  QueryRequest({
    required this.question,
    this.topK = 6,
    required this.subject,
    required this.courseId,
  });

  Map<String, dynamic> toJson() {
    return {
      'question': question,
      'top_k': topK,
      'subject': subject,
      'course_id': courseId,
    };
  }
}

class ContextChunk {
  final String? docId;
  final String? chunkId;
  final int? page;
  final String text;
  final double? score;
  final String? source;

  ContextChunk({
    this.docId,
    this.chunkId,
    this.page,
    required this.text,
    this.score,
    this.source,
  });

  factory ContextChunk.fromJson(Map<String, dynamic> json) {
    return ContextChunk(
      docId: json['doc_id']?.toString(),
      chunkId: json['chunk_id']?.toString(),
      page: _parseInt(json['page']),
      text: json['text']?.toString() ?? '',
      score: _parseDouble(json['score']),
      source: json['source']?.toString(),
    );
  }

  static int? _parseInt(dynamic value) {
    if (value == null) return null;
    if (value is int) return value;
    if (value is double) return value.toInt();
    if (value is String) return int.tryParse(value);
    return null;
  }

  static double? _parseDouble(dynamic value) {
    if (value == null) return null;
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }
}

class QueryResponse {
  final String answer;
  final List<ContextChunk> contexts;
  final int? elapsedMs;

  QueryResponse({
    required this.answer,
    required this.contexts,
    this.elapsedMs,
  });

  factory QueryResponse.fromJson(Map<String, dynamic> json) {
    return QueryResponse(
      answer: json['answer']?.toString() ?? '',
      contexts: (json['contexts'] as List<dynamic>?)
          ?.map((e) => ContextChunk.fromJson(e as Map<String, dynamic>))
          .toList() ??
          [],
      elapsedMs: _parseInt(json['elapsed_ms']),
    );
  }

  static int? _parseInt(dynamic value) {
    if (value == null) return null;
    if (value is int) return value;
    if (value is double) return value.toInt();
    if (value is String) return int.tryParse(value);
    return null;
  }
}

class Flashcard {
  final String question;
  final String answer;
  final String? source;
  final int? page;

  Flashcard({
    required this.question,
    required this.answer,
    this.source,
    this.page,
  });

  Map<String, dynamic> toJson() {
    return {
      'question': question,
      'answer': answer,
      'source': source,
      'page': page,
    };
  }

  factory Flashcard.fromJson(Map<String, dynamic> json) {
    return Flashcard(
      question: json['question']?.toString() ?? '',
      answer: json['answer']?.toString() ?? '',
      source: json['source']?.toString(),
      page: json['page'] as int?,
    );
  }
}