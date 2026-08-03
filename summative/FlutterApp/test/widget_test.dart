// Basic smoke test for the crop yield prediction form. It only checks that
// the UI renders correctly - it does not tap "Predict", since that fires a
// real HTTP call to the live API and would make this test flaky/networked.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:crop_yield_predictor/main.dart';

void main() {
  testWidgets('Crop yield predictor form renders with default values',
      (WidgetTester tester) async {
    await tester.pumpWidget(const CropYieldApp());

    // App bar + submit button.
    expect(find.text('Crop Yield Predictor'), findsOneWidget);
    expect(find.text('Predict'), findsOneWidget);

    // All 6 prediction inputs are present.
    expect(find.byType(TextFormField), findsNWidgets(6));

    // Pre-filled default values (Rwanda/Cassava example row).
    expect(find.text('Rwanda'), findsOneWidget);
    expect(find.text('Cassava'), findsOneWidget);
    expect(find.text('2013'), findsOneWidget);
    expect(find.text('1212'), findsOneWidget);
    expect(find.text('157'), findsOneWidget);
    expect(find.text('19.39'), findsOneWidget);

    // No result/error banner until a prediction has been made.
    expect(find.textContaining('Predicted yield'), findsNothing);
    expect(find.textContaining('Error:'), findsNothing);
  });

  testWidgets('Clearing a required field shows a validation error',
      (WidgetTester tester) async {
    await tester.pumpWidget(const CropYieldApp());

    // Area is the first of the 6 fields.
    await tester.enterText(find.byType(TextFormField).first, '');
    await tester.tap(find.text('Predict'));
    await tester.pump();

    expect(find.text('Area is required'), findsOneWidget);
  });
}
