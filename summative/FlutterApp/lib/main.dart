import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// -----------------------------------------------------------------------
// IMPORTANT: replace this with your deployed Render API base URL, e.g.
//   https://crop-yield-api.onrender.com
// Do NOT include a trailing slash.
// -----------------------------------------------------------------------
const String kApiBaseUrl = "http://127.0.0.1:8000";

void main() {
  runApp(const CropYieldApp());
}

class CropYieldApp extends StatelessWidget {
  const CropYieldApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Yield Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF3D4F7C),
        scaffoldBackgroundColor: const Color(0xFFF6F7FB),
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();

  // One controller per input variable needed for the prediction (6 total):
  // Area, Item, Year, Rainfall, Pesticides, Avg Temperature.
  final _areaController = TextEditingController(text: "Rwanda");
  final _itemController = TextEditingController(text: "Cassava");
  final _yearController = TextEditingController(text: "2013");
  final _rainfallController = TextEditingController(text: "1212");
  final _pesticidesController = TextEditingController(text: "157");
  final _tempController = TextEditingController(text: "19.39");

  bool _isLoading = false;
  String? _resultText;
  bool _isError = false;

  @override
  void dispose() {
    _areaController.dispose();
    _itemController.dispose();
    _yearController.dispose();
    _rainfallController.dispose();
    _pesticidesController.dispose();
    _tempController.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _resultText = null;
      _isError = false;
    });

    final Uri url = Uri.parse("$kApiBaseUrl/predict");

    final Map<String, dynamic> body = {
      "area": _areaController.text.trim(),
      "item": _itemController.text.trim(),
      "year": int.tryParse(_yearController.text.trim()) ?? 0,
      "average_rain_fall_mm_per_year":
          double.tryParse(_rainfallController.text.trim()) ?? 0.0,
      "pesticides_tonnes":
          double.tryParse(_pesticidesController.text.trim()) ?? 0.0,
      "avg_temp": double.tryParse(_tempController.text.trim()) ?? 0.0,
    };

    try {
      final response = await http
          .post(
            url,
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final yieldHgHa = data["predicted_yield_hg_per_ha"];
        final yieldTonnes = data["predicted_yield_tonnes_per_ha"];
        setState(() {
          _resultText =
              "Predicted yield: $yieldHgHa hg/ha  (≈ $yieldTonnes tonnes/ha)";
          _isError = false;
        });
      } else {
        // Surfaces FastAPI/Pydantic validation errors (missing fields,
        // out-of-range values, unknown Area/Item) directly to the user.
        final data = jsonDecode(response.body);
        setState(() {
          _resultText = "Error: ${_extractErrorMessage(data)}";
          _isError = true;
        });
      }
    } catch (e) {
      setState(() {
        _resultText = "Network error: could not reach the API. $e";
        _isError = true;
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  String _extractErrorMessage(dynamic data) {
    if (data is Map && data["detail"] is List && data["detail"].isNotEmpty) {
      final first = data["detail"][0];
      final loc = (first["loc"] as List?)?.join(" -> ") ?? "";
      return "${first["msg"]} ($loc)";
    }
    if (data is Map && data["detail"] is String) {
      return data["detail"];
    }
    return "Unexpected error response.";
  }

  String? _requiredValidator(String? value, {String label = "This field"}) {
    if (value == null || value.trim().isEmpty) {
      return "$label is required";
    }
    return null;
  }

  String? _numberValidator(String? value, {required String label, bool isInt = false}) {
    final requiredError = _requiredValidator(value, label: label);
    if (requiredError != null) return requiredError;
    final parsed = isInt ? int.tryParse(value!.trim()) : double.tryParse(value!.trim());
    if (parsed == null) {
      return "$label must be a valid number";
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Crop Yield Predictor"),
        backgroundColor: const Color(0xFF3D4F7C),
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  "Predict crop yield (hg/ha) from country, crop, and climate inputs.",
                  style: TextStyle(fontSize: 14, color: Colors.black54),
                ),
                const SizedBox(height: 20),

                _buildTextField(
                  controller: _areaController,
                  label: "Area (country)",
                  hint: "e.g. Rwanda",
                  validator: (v) => _requiredValidator(v, label: "Area"),
                ),
                const SizedBox(height: 14),

                _buildTextField(
                  controller: _itemController,
                  label: "Item (crop)",
                  hint: "e.g. Cassava, Maize, Potatoes, Rice, paddy",
                  validator: (v) => _requiredValidator(v, label: "Item"),
                ),
                const SizedBox(height: 14),

                _buildTextField(
                  controller: _yearController,
                  label: "Year",
                  hint: "e.g. 2013",
                  keyboardType: TextInputType.number,
                  validator: (v) => _numberValidator(v, label: "Year", isInt: true),
                ),
                const SizedBox(height: 14),

                _buildTextField(
                  controller: _rainfallController,
                  label: "Average rainfall (mm/year)",
                  hint: "e.g. 1212",
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                  validator: (v) => _numberValidator(v, label: "Rainfall"),
                ),
                const SizedBox(height: 14),

                _buildTextField(
                  controller: _pesticidesController,
                  label: "Pesticides used (tonnes)",
                  hint: "e.g. 157",
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                  validator: (v) => _numberValidator(v, label: "Pesticides"),
                ),
                const SizedBox(height: 14),

                _buildTextField(
                  controller: _tempController,
                  label: "Average temperature (°C)",
                  hint: "e.g. 19.39",
                  keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
                  validator: (v) => _numberValidator(v, label: "Temperature"),
                ),
                const SizedBox(height: 24),

                SizedBox(
                  height: 50,
                  child: ElevatedButton(
                    onPressed: _isLoading ? null : _predict,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF3D4F7C),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: _isLoading
                        ? const SizedBox(
                            width: 22,
                            height: 22,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                              strokeWidth: 2.5,
                            ),
                          )
                        : const Text("Predict", style: TextStyle(fontSize: 16)),
                  ),
                ),
                const SizedBox(height: 24),

                if (_resultText != null)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: _isError ? const Color(0xFFFCE8E8) : const Color(0xFFE8F0E8),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: _isError ? Colors.redAccent : Colors.green,
                      ),
                    ),
                    child: Text(
                      _resultText!,
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: _isError ? Colors.red.shade800 : Colors.green.shade800,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    TextInputType keyboardType = TextInputType.text,
    required String? Function(String?) validator,
  }) {
    return TextFormField(
      controller: controller,
      keyboardType: keyboardType,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      ),
      validator: validator,
    );
  }
}
