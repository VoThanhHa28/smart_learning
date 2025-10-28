import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'dart:async';
import '../../../services/streaming_api_service.dart';
import '../../../models/query_models.dart';

class QueryScreen extends StatefulWidget {
  final String subject;
  final String courseId;
  final String docId;

  const QueryScreen({
    super.key,
    required this.subject,
    required this.courseId,
    required this.docId,
  });

  @override
  State<QueryScreen> createState() => _QueryScreenState();
}

class _QueryScreenState extends State<QueryScreen> with SingleTickerProviderStateMixin {
  final StreamingApiService _streamingApi = StreamingApiService();
  final TextEditingController _questionController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  // ✅ FIX: Use List<ChatMessage> instead of List<Map>
  final List<ChatMessage> _chatHistory = [];

  bool _isStreaming = false;
  StreamSubscription? _streamSubscription;
  User? _currentUser;
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _checkUserAuth();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
  }

  void _checkUserAuth() {
    _currentUser = FirebaseAuth.instance.currentUser;

    if (_currentUser == null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          _showError('Vui lòng đăng nhập để sử dụng tính năng này');
          Navigator.pop(context);
        }
      });
    } else {
      print('✅ User: ${_currentUser!.email}');
      print('📚 Course: ${widget.subject} (${widget.courseId})');
    }
  }

  @override
  void dispose() {
    _questionController.dispose();
    _scrollController.dispose();
    _streamSubscription?.cancel();
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _submitQuery() async {
    if (_currentUser == null) {
      _showError('User not logged in');
      return;
    }

    final question = _questionController.text.trim();
    if (question.isEmpty) {
      _showError('Vui lòng nhập câu hỏi');
      return;
    }

    // Cancel previous stream
    await _streamSubscription?.cancel();

    // Add user message to chat
    setState(() {
      _chatHistory.add(ChatMessage(
        text: question,
        isUser: true,
        timestamp: DateTime.now(),
      ));
      _isStreaming = true;
    });

    _questionController.clear();
    _scrollToBottom();
    _animationController.repeat();

    // Add AI message placeholder
    final aiMessageIndex = _chatHistory.length;
    setState(() {
      _chatHistory.add(ChatMessage(
        text: '',
        isUser: false,
        timestamp: DateTime.now(),
        isStreaming: true,
      ));
    });

    try {
      final request = QueryRequest(
        question: question,
        topK: 6,
        subject: widget.subject,
        courseId: widget.courseId,
      );

      print('');
      print('='*80);
      print('📤 Submitting Query');
      print('='*80);
      print('Question: $question');
      print('Subject: ${widget.subject}');
      print('Course: ${widget.courseId}');
      print('User: ${_currentUser!.uid}');
      print('='*80);

      final startTime = DateTime.now();
      String fullAnswer = '';

      _streamSubscription = _streamingApi.queryStream(request).listen(
            (chunk) {
          fullAnswer += chunk;

          setState(() {
            _chatHistory[aiMessageIndex] = ChatMessage(
              text: fullAnswer,
              isUser: false,
              timestamp: startTime,
              isStreaming: true,
            );
          });

          _scrollToBottom();
        },
        onDone: () {
          final elapsed = DateTime.now().difference(startTime);

          print('');
          print('✅ Streaming completed');
          print('   Time: ${elapsed.inSeconds}s');
          print('   Answer length: ${fullAnswer.length} chars');
          print('');

          setState(() {
            _chatHistory[aiMessageIndex] = ChatMessage(
              text: fullAnswer,
              isUser: false,
              timestamp: startTime,
              isStreaming: false,
              elapsedMs: elapsed.inMilliseconds,
            );
            _isStreaming = false;
          });

          _animationController.stop();
          _scrollToBottom();

          _showSuccess('Hoàn thành sau ${elapsed.inSeconds}s');
        },
        onError: (error) {
          print('');
          print('❌ Stream error: $error');
          print('');

          setState(() {
            _chatHistory[aiMessageIndex] = ChatMessage(
              text: 'Lỗi: $error',
              isUser: false,
              timestamp: DateTime.now(),
              isStreaming: false,
              isError: true,
            );
            _isStreaming = false;
          });

          _animationController.stop();
          _showError('Lỗi: $error');
        },
      );
    } catch (e, stackTrace) {
      print('');
      print('❌ Query failed: $e');
      print('Stack: $stackTrace');
      print('');

      setState(() {
        if (_chatHistory.length > aiMessageIndex) {
          _chatHistory[aiMessageIndex] = ChatMessage(
            text: 'Lỗi: $e',
            isUser: false,
            timestamp: DateTime.now(),
            isStreaming: false,
            isError: true,
          );
        }
        _isStreaming = false;
      });

      _animationController.stop();
      _showError('Lỗi: $e');
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _clearChat() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Xóa lịch sử chat'),
        content: const Text('Bạn có chắc muốn xóa tất cả tin nhắn?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Hủy'),
          ),
          TextButton(
            onPressed: () {
              setState(() {
                _chatHistory.clear();
              });
              Navigator.pop(context);
              _showSuccess('Đã xóa lịch sử chat');
            },
            child: const Text('Xóa', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.white),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.red.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  void _showSuccess(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.check_circle_outline, color: Colors.white),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.green.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _capitalize(widget.subject),
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            if (_currentUser != null)
              Text(
                _currentUser!.email ?? 'No email',
                style: TextStyle(
                  fontSize: 11,
                  color: Colors.grey[600],
                  fontWeight: FontWeight.normal,
                ),
              ),
          ],
        ),
        actions: [
          if (_chatHistory.isNotEmpty)
            IconButton(
              icon: Badge(
                label: Text('${_chatHistory.where((m) => m.isUser).length}'),
                child: const Icon(Icons.chat_bubble_outline),
              ),
              onPressed: () {},
              tooltip: '${_chatHistory.where((m) => m.isUser).length} câu hỏi',
            ),
          if (_chatHistory.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              onPressed: _clearChat,
              tooltip: 'Xóa lịch sử',
            ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(
            height: 1,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.blue.shade200,
                  Colors.purple.shade200,
                ],
              ),
            ),
          ),
        ),
      ),
      body: Column(
        children: [
          // Course info banner
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.03),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.blue.shade400, Colors.purple.shade400],
                    ),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(
                    Icons.school,
                    color: Colors.white,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.courseId,
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Text(
                        'Smart Learning AI Assistant',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // Chat messages
          Expanded(
            child: _chatHistory.isEmpty
                ? _buildEmptyState()
                : ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _chatHistory.length,
              itemBuilder: (context, index) {
                return _buildMessageBubble(_chatHistory[index]);
              },
            ),
          ),

          // Input area
          _buildInputArea(),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.blue.shade100, Colors.purple.shade100],
                ),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.chat_bubble_outline,
                size: 64,
                color: Colors.blue.shade700,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Bắt đầu cuộc trò chuyện',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.grey[800],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Đặt câu hỏi về tài liệu học tập của bạn',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            _buildSuggestedQuestions(),
          ],
        ),
      ),
    );
  }

  Widget _buildSuggestedQuestions() {
    final suggestions = [
      {'icon': '💡', 'text': 'Tóm tắt nội dung chính'},
      {'icon': '📝', 'text': 'Giải thích khái niệm'},
      {'icon': '❓', 'text': 'Câu hỏi ôn tập'},
    ];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: suggestions.map((item) {
        return InkWell(
          onTap: () {
            _questionController.text = item['text']!;
          },
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.blue.shade50,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.blue.shade200),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(item['icon']!, style: const TextStyle(fontSize: 16)),
                const SizedBox(width: 8),
                Text(
                  item['text']!,
                  style: TextStyle(
                    color: Colors.blue.shade700,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildMessageBubble(ChatMessage message) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment:
        message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          if (!message.isUser) ...[
            CircleAvatar(
              radius: 16,
              backgroundColor: Colors.blue.shade100,
              child: Icon(
                Icons.smart_toy,
                size: 18,
                color: Colors.blue.shade700,
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                gradient: message.isUser
                    ? LinearGradient(
                  colors: [Colors.blue.shade500, Colors.purple.shade500],
                )
                    : null,
                color: message.isUser
                    ? null
                    : (message.isError ? Colors.red.shade50 : Colors.white),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SelectableText(
                    message.text.isEmpty ? 'Đang suy nghĩ...' : message.text,
                    style: TextStyle(
                      fontSize: 15,
                      height: 1.4,
                      color: message.isUser ? Colors.white : Colors.black87,
                    ),
                  ),
                  if (message.isStreaming)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 12,
                            height: 12,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                Colors.blue.shade400,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'Đang trả lời...',
                            style: TextStyle(
                              fontSize: 11,
                              color: Colors.grey[600],
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ],
                      ),
                    ),
                  if (!message.isUser && !message.isStreaming && message.elapsedMs != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        '⏱️ ${(message.elapsedMs! / 1000).toStringAsFixed(1)}s',
                        style: TextStyle(
                          fontSize: 10,
                          color: Colors.grey[500],
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
          if (message.isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 16,
              backgroundColor: Colors.grey[300],
              backgroundImage: _currentUser?.photoURL != null
                  ? NetworkImage(_currentUser!.photoURL!)
                  : null,
              child: _currentUser?.photoURL == null
                  ? Icon(Icons.person, size: 18, color: Colors.grey[600])
                  : null,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildInputArea() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -5),
          ),
        ],
      ),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  borderRadius: BorderRadius.circular(25),
                ),
                child: TextField(
                  controller: _questionController,
                  decoration: InputDecoration(
                    hintText: 'Nhập câu hỏi của bạn...',
                    hintStyle: TextStyle(color: Colors.grey[500]),
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 12,
                    ),
                  ),
                  maxLines: null,
                  textCapitalization: TextCapitalization.sentences,
                  enabled: !_isStreaming,
                  onSubmitted: (_) => _submitQuery(),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: _isStreaming
                      ? [Colors.grey.shade400, Colors.grey.shade500]
                      : [Colors.blue.shade500, Colors.purple.shade500],
                ),
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: _isStreaming
                        ? Colors.grey.withOpacity(0.3)
                        : Colors.blue.withOpacity(0.3),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: _isStreaming ? null : _submitQuery,
                  borderRadius: BorderRadius.circular(25),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    child: Icon(
                      _isStreaming ? Icons.stop_circle : Icons.send,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _capitalize(String text) {
    if (text.isEmpty) return text;
    final normalized = text.replaceAll('_', ' ');
    return normalized.split(' ').map((word) {
      if (word.isEmpty) return word;
      return word[0].toUpperCase() + word.substring(1).toLowerCase();
    }).join(' ');
  }
}

// ✅ Chat message model - Place at bottom of file
class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final bool isStreaming;
  final bool isError;
  final int? elapsedMs;

  ChatMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.isStreaming = false,
    this.isError = false,
    this.elapsedMs,
  });
}