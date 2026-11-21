import 'dart:convert';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:garment_productivity_predictor/main.dart';

Future<void> fill(WidgetTester t, List<String> v) async {
  final f = find.byType(TextField);
  for (var i = 0; i < v.length; i++) {
    await t.enterText(f.at(i), v[i]);
  }
}

void main() {
  final good = ['SEWING', 'tuesday', '1', '2', '0.75', '22.5', '900', '5000', '0', '0', '0', '40'];

  testWidgets('success path shows the prediction and the sent JSON is right', (tester) async {
    late Map<String, dynamic> sent;
    await http.runWithClient(() async {
      await tester.pumpWidget(const GarmentProductivityApp());
      await fill(tester, good);
      await tester.ensureVisible(find.text('Predict'));
      await tester.tap(find.text('Predict'));
      await tester.pumpAndSettle();
    }, () => MockClient((req) async {
          sent = jsonDecode(req.body) as Map<String, dynamic>;
          return http.Response(jsonEncode({'predicted_productivity': 0.665, 'message': 'Predicted productivity 0.67 vs target 0.75: shortfall.'}), 200,
              headers: {'content-type': 'application/json'});
        }));
    expect(find.byKey(const Key('predicted_value')), findsOneWidget);
    expect(find.text('0.67'), findsOneWidget);
    expect(sent['department'], 'sewing');
    expect(sent['day'], 'Tuesday');
    expect(sent['team'], 1);
    expect(sent['targeted_productivity'], 0.75);
    expect(sent['no_of_workers'], 40.0);
  });

  testWidgets('422 from the API is shown as readable lines', (tester) async {
    await http.runWithClient(() async {
      await tester.pumpWidget(const GarmentProductivityApp());
      await fill(tester, good);
      await tester.ensureVisible(find.text('Predict'));
      await tester.tap(find.text('Predict'));
      await tester.pumpAndSettle();
    }, () => MockClient((req) async => http.Response(
        jsonEncode({'detail': [{'loc': ['body', 'smv'], 'msg': 'Input should be less than or equal to 100'}]}), 422,
        headers: {'content-type': 'application/json'})));
    expect(find.textContaining('smv: Input should be less than or equal to 100'), findsOneWidget);
  });

  testWidgets('no connection shows a readable message', (tester) async {
    await http.runWithClient(() async {
      await tester.pumpWidget(const GarmentProductivityApp());
      await fill(tester, good);
      await tester.ensureVisible(find.text('Predict'));
      await tester.tap(find.text('Predict'));
      await tester.pumpAndSettle();
    }, () => MockClient((req) async => throw http.ClientException('boom')));
    expect(find.textContaining('Network error'), findsOneWidget);
  });

  testWidgets('no overflow on a small phone with big text', (tester) async {
    tester.view.physicalSize = const Size(720, 1280);
    tester.view.devicePixelRatio = 2.0;
    tester.platformDispatcher.textScaleFactorTestValue = 1.3;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(const GarmentProductivityApp());
    await tester.ensureVisible(find.text('Predict'));
    await tester.tap(find.text('Predict'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });
}
