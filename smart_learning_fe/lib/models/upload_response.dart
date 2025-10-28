class UploadResponse {
  final String docId;
  final String courseId;
  final String? userId; // ✅ Thêm userId
  final int? pages;
  final int? chunks;
  final int? elapsedMs;
  final List<String> warnings;

  UploadResponse({
    required this.docId,
    required this.courseId,
    this.userId, // ✅ Thêm vào constructor
    this.pages,
    this.chunks,
    this.elapsedMs,
    this.warnings = const [],
  });

  factory UploadResponse.fromJson(Map<String, dynamic> json) {
    try {
      return UploadResponse(
        docId: json['doc_id']?.toString() ?? '',
        courseId: json['course_id']?.toString() ?? '',
        userId: json['user_id']?.toString(), // ✅ Parse userId
        pages: _parseInt(json['pages']),
        chunks: _parseInt(json['chunks']),
        elapsedMs: _parseInt(json['elapsed_ms']),
        warnings: _parseStringList(json['warnings']),
      );
    } catch (e) {
      print('❌ Error parsing UploadResponse: $e');
      print('JSON data: $json');
      rethrow;
    }
  }

  static int? _parseInt(dynamic value) {
    if (value == null) return null;
    if (value is int) return value;
    if (value is double) return value.toInt();
    if (value is String) return int.tryParse(value);
    return null;
  }

  static List<String> _parseStringList(dynamic value) {
    if (value == null) return [];
    if (value is List) {
      return value.map((e) => e?.toString() ?? '').toList();
    }
    return [];
  }

  Map<String, dynamic> toJson() {
    return {
      'doc_id': docId,
      'course_id': courseId,
      'user_id': userId, // ✅ Thêm vào JSON
      'pages': pages,
      'chunks': chunks,
      'elapsed_ms': elapsedMs,
      'warnings': warnings,
    };
  }

  @override
  String toString() {
    return 'UploadResponse(docId: $docId, courseId: $courseId, userId: $userId, chunks: $chunks)';
  }
}