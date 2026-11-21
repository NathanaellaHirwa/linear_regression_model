import 'dart:async';
import 'dart:convert';
import 'dart:io';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Base URL of the deployed FastAPI service (Render). Change it here, or override it without editing the code:
///   flutter run --dart-define=API_BASE_URL=https://your-service.onrender.com
/// Android emulator talking to an API on your own PC:  --dart-define=API_BASE_URL=http://10.0.2.2:8000
const String kApiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'https://garment-productivity-api.onrender.com',
);

void main() => runApp(const GarmentProductivityApp());

// ---------------------------------------------------------------------------
// The 12 inputs of the model. Names, types and ranges mirror the Pydantic
// model (TeamDayInput) in API/prediction.py.
// ---------------------------------------------------------------------------
enum FieldKind { integer, decimal, choice }

class FieldSpec {
  const FieldSpec({
    required this.key,
    required this.label,
    required this.hint,
    required this.kind,
    this.min = 0,
    this.max = 0,
    this.choices = const <String>[],
  });

  final String key; // JSON field name expected by the API
  final String label; // label shown above the text field
  final String hint; // short helper text: allowed values
  final FieldKind kind;
  final num min;
  final num max;
  final List<String> choices;
}

const List<FieldSpec> kFields = <FieldSpec>[
  FieldSpec(key: 'department', label: 'Department', hint: 'sewing / finishing', kind: FieldKind.choice, choices: ['sewing', 'finishing']),
  FieldSpec(key: 'day', label: 'Day', hint: 'Saturday ... Thursday', kind: FieldKind.choice, choices: ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']),
  FieldSpec(key: 'team', label: 'Team', hint: '1 - 12', kind: FieldKind.integer, min: 1, max: 12),
  FieldSpec(key: 'week_of_month', label: 'Week of month', hint: '1 - 5', kind: FieldKind.integer, min: 1, max: 5),
  FieldSpec(key: 'targeted_productivity', label: 'Target productivity', hint: '0.05 - 1.0', kind: FieldKind.decimal, min: 0.05, max: 1.0),
  FieldSpec(key: 'smv', label: 'SMV (min/garment)', hint: '1 - 100', kind: FieldKind.decimal, min: 1, max: 100),
  FieldSpec(key: 'wip', label: 'Work in progress', hint: '0 - 30000', kind: FieldKind.integer, min: 0, max: 30000),
  FieldSpec(key: 'over_time', label: 'Overtime (min)', hint: '0 - 30000', kind: FieldKind.integer, min: 0, max: 30000),
  FieldSpec(key: 'incentive', label: 'Incentive', hint: '0 - 5000', kind: FieldKind.integer, min: 0, max: 5000),
  FieldSpec(key: 'idle_men', label: 'Idle workers', hint: '0 - 50', kind: FieldKind.integer, min: 0, max: 50),
  FieldSpec(key: 'no_of_style_change', label: 'Style changes', hint: '0 - 5', kind: FieldKind.integer, min: 0, max: 5),
  FieldSpec(key: 'no_of_workers', label: 'Number of workers', hint: '1 - 100', kind: FieldKind.decimal, min: 1, max: 100),
];

// ---------------------------------------------------------------------------
// What the display area can show.
// ---------------------------------------------------------------------------
enum ResultStatus { idle, loading, success, error }

class PredictionResult {
  const PredictionResult.idle()
      : status = ResultStatus.idle,
        value = null,
        lines = const <String>['Fill in the 12 fields above and press Predict.'];
  const PredictionResult.loading()
      : status = ResultStatus.loading,
        value = null,
        lines = const <String>[
          'Contacting the server...',
          'The first request can take up to a minute while the free server wakes up.',
        ];
  const PredictionResult.success(this.value, this.lines) : status = ResultStatus.success;
  const PredictionResult.error(this.lines)
      : status = ResultStatus.error,
        value = null;

  final ResultStatus status;
  final double? value;
  final List<String> lines;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
class GarmentProductivityApp extends StatelessWidget {
  const GarmentProductivityApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Garment Productivity Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF8A4F7D)), useMaterial3: true),
      home: const PredictPage(),
    );
  }
}

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final f in kFields) f.key: TextEditingController(),
  };
  final GlobalKey _resultKey = GlobalKey();
  PredictionResult _result = const PredictionResult.idle();
  bool _loading = false;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  /// Checks every field on the phone first and returns readable messages.
  /// Missing value, wrong type and out-of-range value are all reported here;
  /// the API repeats the same checks, so it is never trusted blindly.
  ({Map<String, Object> payload, List<String> errors}) _validate() {
    final payload = <String, Object>{};
    final errors = <String>[];
    for (final f in kFields) {
      final text = _controllers[f.key]!.text.trim();
      if (text.isEmpty) {
        errors.add('${f.label} is required (${f.hint}).');
        continue;
      }
      switch (f.kind) {
        case FieldKind.integer:
          final value = int.tryParse(text);
          if (value == null) {
            errors.add('${f.label} must be a whole number such as ${f.min}, not "$text".');
          } else if (value < f.min || value > f.max) {
            errors.add('${f.label} must be between ${f.min} and ${f.max} (you entered $value).');
          } else {
            payload[f.key] = value;
          }
        case FieldKind.decimal:
          final value = double.tryParse(text.replaceAll(',', '.'));
          if (value == null || value.isNaN || value.isInfinite) {
            errors.add('${f.label} must be a number such as ${f.min}, not "$text".');
          } else if (value < f.min || value > f.max) {
            errors.add('${f.label} must be between ${f.min} and ${f.max} (you entered $text).');
          } else {
            payload[f.key] = value;
          }
        case FieldKind.choice:
          final value = f.choices.firstWhere(
            (c) => c.toLowerCase() == text.toLowerCase(),
            orElse: () => '',
          );
          if (value.isEmpty) {
            errors.add('${f.label} must be one of: ${f.choices.join(', ')} (you entered "$text").');
          } else {
            payload[f.key] = value;
          }
      }
    }
    return (payload: payload, errors: errors);
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    final check = _validate();
    if (check.errors.isNotEmpty) {
      _show(PredictionResult.error(check.errors));
      return;
    }

    setState(() => _loading = true);
    _show(const PredictionResult.loading());
    try {
      final response = await http
          .post(
            Uri.parse('$kApiBaseUrl/predict'),
            headers: const {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: jsonEncode(check.payload),
          )
          .timeout(const Duration(seconds: 90));
      _show(_interpret(response));
    } on TimeoutException {
      _show(const PredictionResult.error([
        'The server did not answer in time.',
        'A free Render server sleeps after 15 minutes without traffic and can need about a minute to wake up. Please try again.',
      ]));
    } on SocketException {
      _show(const PredictionResult.error([
        'Could not reach the server.',
        'Check the internet connection of the phone or emulator and try again.',
      ]));
    } on http.ClientException catch (e) {
      _show(PredictionResult.error(['Network error: ${e.message}']));
    } catch (e) {
      _show(PredictionResult.error(['Unexpected error: $e']));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  /// Turns an HTTP response into something a teacher can read.
  PredictionResult _interpret(http.Response response) {
    dynamic body;
    try {
      body = jsonDecode(utf8.decode(response.bodyBytes));
    } on FormatException {
      body = null;
    }

    if (response.statusCode == 200 && body is Map && body['predicted_productivity'] is num) {
      final value = (body['predicted_productivity'] as num).toDouble();
      final message = body['message'] is String ? body['message'] as String : 'Prediction received.';
      return PredictionResult.success(value, [message]);
    }

    if (response.statusCode == 422 && body is Map) {
      final detail = body['detail'];
      if (detail is List) {
        // FastAPI validation errors: [{loc: [body, age], msg: "..."}, ...]
        final lines = <String>[];
        for (final item in detail) {
          if (item is Map) {
            final loc = item['loc'] is List ? (item['loc'] as List).where((p) => p != 'body').join('.') : '';
            lines.add('${loc.isEmpty ? 'Input' : loc}: ${item['msg']}');
          }
        }
        return PredictionResult.error(['The server rejected the input:', ...lines]);
      }
      if (detail is String) return PredictionResult.error(['The server rejected the input: $detail']);
    }
    return PredictionResult.error(['The server returned an error (HTTP ${response.statusCode}). Please try again later.']);
  }

  void _show(PredictionResult result) {
    if (!mounted) return;
    setState(() => _result = result);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final ctx = _resultKey.currentContext;
      if (ctx != null) {
        Scrollable.ensureVisible(ctx, duration: const Duration(milliseconds: 250), alignment: 0.5);
      }
    });
  }

  Widget _field(FieldSpec f, bool isLast) {
    return TextField(
      controller: _controllers[f.key],
      keyboardType: f.kind == FieldKind.choice
          ? TextInputType.text
          : const TextInputType.numberWithOptions(signed: true, decimal: true),
      textInputAction: isLast ? TextInputAction.done : TextInputAction.next,
      autocorrect: false,
      decoration: InputDecoration(
        labelText: f.label,
        helperText: f.hint,
        border: const OutlineInputBorder(),
        isDense: true,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Two labelled columns of text fields, aligned in rows.
    final rows = <Widget>[];
    for (var i = 0; i < kFields.length; i += 2) {
      final hasSecond = i + 1 < kFields.length;
      rows.add(Padding(
        padding: const EdgeInsets.only(bottom: 12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(child: _field(kFields[i], false)),
            const SizedBox(width: 12),
            Expanded(child: hasSecond ? _field(kFields[i + 1], i + 1 == kFields.length - 1) : const SizedBox.shrink()),
          ],
        ),
      ));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Garment Productivity Predictor')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                "Enter the team's details for the day (known before the shift) to predict its actual productivity (1.0 = 100% of planned output).",
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              ...rows,
              const SizedBox(height: 4),
              SizedBox(
                height: 48,
                child: FilledButton(onPressed: _loading ? null : _predict, child: const Text('Predict')),
              ),
              const SizedBox(height: 16),
              _ResultCard(key: _resultKey, result: _result),
            ],
          ),
        ),
      ),
    );
  }
}

/// The single display area: shows the prediction, the progress message or a readable error.
class _ResultCard extends StatelessWidget {
  const _ResultCard({super.key, required this.result});

  final PredictionResult result;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final isError = result.status == ResultStatus.error;
    final isSuccess = result.status == ResultStatus.success;
    final background = isError ? scheme.errorContainer : (isSuccess ? scheme.primaryContainer : scheme.surfaceContainerHighest);
    final foreground = isError ? scheme.onErrorContainer : (isSuccess ? scheme.onPrimaryContainer : scheme.onSurface);

    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            isSuccess ? 'Predicted productivity' : (isError ? 'Please check your input' : 'Result'),
            style: Theme.of(context).textTheme.labelLarge?.copyWith(color: foreground),
          ),
          const SizedBox(height: 8),
          if (result.status == ResultStatus.loading) ...[
            const LinearProgressIndicator(),
            const SizedBox(height: 12),
          ],
          if (isSuccess)
            Text(
              result.value!.toStringAsFixed(2),
              key: const Key('predicted_value'),
              style: Theme.of(context).textTheme.displaySmall?.copyWith(color: foreground, fontWeight: FontWeight.bold),
            ),
          for (final line in result.lines)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(isError && !line.endsWith(':') && result.lines.length > 1 && line != result.lines.first ? '• $line' : line,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: foreground)),
            ),
        ],
      ),
    );
  }
}
