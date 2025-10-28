class Course {
  final String courseId;
  final String subject;

  Course({
    required this.courseId,
    required this.subject,
  });

  factory Course.fromJson(Map<String, dynamic> json) {
    return Course(
      courseId: json['course_id'] as String,
      subject: json['subject'] as String,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'course_id': courseId,
      'subject': subject,
    };
  }
}