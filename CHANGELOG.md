# Changelog

## 0.6.1

Bug fixes:
- Fix flaky `ModelInsight` tests [#395](https://github.com/salesforce/TransmogrifAI/pull/395)
- Avoid creating `SparseVector`s for LOCO [#377](https://github.com/salesforce/TransmogrifAI/pull/377)

New features / updates:
- Model combiner [#385](https://github.com/salesforce/TransmogrifAI/pull/399)
- Added new sample for HousingPrices [#365](https://github.com/salesforce/TransmogrifAI/pull/365)
- Test to verify that custom metrics appear in model insight metrics [#387](https://github.com/salesforce/TransmogrifAI/pull/387)
- Add `FeatureDistribution` to `SerializationFormat`s [#383](https://github.com/salesforce/TransmogrifAI/pull/383)
- Add metadata to `OpStandadrdScaler` to allow for descaling [#378](https://github.com/salesforce/TransmogrifAI/pull/378)
- Improve json serde error in `evalMetFromJson` [#380](https://github.com/salesforce/TransmogrifAI/pull/380)
- Track mean & standard deviation of text length as a metric for text feature [#354](https://github.com/salesforce/TransmogrifAI/pull/354)
- Making model selectors robust to failing models [#372](https://github.com/salesforce/TransmogrifAI/pull/372)
- Use compact and compressed model json by default [#375](https://github.com/salesforce/TransmogrifAI/pull/375)
- Descale feature contribution for Linear Regression & Logistic Regression [#345](https://github.com/salesforce/TransmogrifAI/pull/345)

Dependency updates:   
- Update tika version [#382](https://github.com/salesforce/TransmogrifAI/pull/382)
- Revert back to Spark 2.3 [#399](https://github.com/salesforce/TransmogrifAI/pull/399)

## 0.6.0

Bug fixes:
- Quick Fix Alias Type Names [#346](https://github.com/salesforce/TransmogrifAI/pull/346)
- Forecast Evaluator - fixes SMAPE, adds MASE and Seasonal Error metrics [#342](https://github.com/salesforce/TransmogrifAI/pull/342)

New features / updates:
- Aggregate LOCOs of DateToUnitCircleTransformer. [#349](https://github.com/salesforce/TransmogrifAI/pull/349)
- Convert lambda functions into concrete classes to allow compatibility with Scala 2.12 [#357](https://github.com/salesforce/TransmogrifAI/pull/357)
- Replace mapValues with immutable Map where applicable [#363](https://github.com/salesforce/TransmogrifAI/pull/363)
- Aggregate spark metrics during run time instead of post processing by default [#358](https://github.com/salesforce/TransmogrifAI/pull/358)
- Allow customizing serialization for FeatureGenerator extract function [#352](https://github.com/salesforce/TransmogrifAI/pull/352)
- Update helloworld examples to be simple [#351](https://github.com/salesforce/TransmogrifAI/pull/351)
- Adding `key` ctor field in all RawFeatureFilter results [#348](https://github.com/salesforce/TransmogrifAI/pull/348)
- Forecast evaluator + SMAPE metric [#337](https://github.com/salesforce/TransmogrifAI/pull/337)
- Local scoring for model with features of all types [#340](https://github.com/salesforce/TransmogrifAI/pull/340)
- Remove local runner + update docs [#335](https://github.com/salesforce/TransmogrifAI/pull/335)
- Added missing test for java conversions [#334](https://github.com/salesforce/TransmogrifAI/pull/334)
- Get rid of scalaj-collections [#333](https://github.com/salesforce/TransmogrifAI/pull/333)
- Workflow independent model loading [#274](https://github.com/salesforce/TransmogrifAI/pull/274)
- Aggregated LOCOs of SmartTextVectorizer outputs [#308](https://github.com/salesforce/TransmogrifAI/pull/308)
- Added community projects docs section [#326](https://github.com/salesforce/TransmogrifAI/pull/326)
- Add FeatureBuilder.fromSchema [#325](https://github.com/salesforce/TransmogrifAI/pull/325)
- Improve WeekOfMonth in date transformers [#323](https://github.com/salesforce/TransmogrifAI/pull/323)
- Improved datetime unit transformer shortcuts - Part 2 [#319](https://github.com/salesforce/TransmogrifAI/pull/319)
- Correctly pass main class for CLI sub project [#321](https://github.com/salesforce/TransmogrifAI/pull/321)
- Serialize blacklisted map keys with the model + updated access on workflow/model members [#320](https://github.com/salesforce/TransmogrifAI/pull/320)
- Improved datetime unit transformer shortcuts [#316](https://github.com/salesforce/TransmogrifAI/pull/316)
- Improved OpScalarStandardScalerTest [#317](https://github.com/salesforce/TransmogrifAI/pull/317)
- improved PercentileCalibratorTest [#318](https://github.com/salesforce/TransmogrifAI/pull/318)
- Added concrete wrappers for HashingTF, NGram and StopWordsRemover [#314](https://github.com/salesforce/TransmogrifAI/pull/314)
- Avoid singleton random generators [#312](https://github.com/salesforce/TransmogrifAI/pull/312)
- Remove free function aggregation with feature builders [#311](https://github.com/salesforce/TransmogrifAI/pull/311)
- Added util methods to create class/object by name + retrieve type tag by type name [#310](https://github.com/salesforce/TransmogrifAI/pull/310)

 Dependency updates:
- Bump shadowjar plugin to 5.0.0 [#306](https://github.com/salesforce/TransmogrifAI/pull/306)
- Bump Apache Tika to 1.21 [#331](https://github.com/salesforce/TransmogrifAI/pull/331)
- Enable CicleCI version 2.1 [#353](https://github.com/salesforce/TransmogrifAI/pull/353)

## 0.5.3

Bug fixes:
- Threshold metrics calculation fix when unseen labels are present [#293](https://github.com/salesforce/TransmogrifAI/pull/293)
- DataCutter-related fixes for multiclass [#263](https://github.com/salesforce/TransmogrifAI/pull/263)
- Fixed onSetInput so is always called with new input [#280](https://github.com/salesforce/TransmogrifAI/pull/280)

New features / updates:
- Improved test SmartTextMapVectorizerTest [#296](https://github.com/salesforce/TransmogrifAI/pull/296)
- Add check to raw feature filter for removing all features [#303](https://github.com/salesforce/TransmogrifAI/pull/303)
- Spec-ifying ngram similarity tests [#299](https://github.com/salesforce/TransmogrifAI/pull/299)
- Add random test feature generator to generate datasets with features of *all* types [#298](https://github.com/salesforce/TransmogrifAI/pull/298)
- Spec-ifying NGramTest [#297](https://github.com/salesforce/TransmogrifAI/pull/297)
- Added base spec for testing Spark wrapping transformers [#295](https://github.com/salesforce/TransmogrifAI/pull/295)
- Add/upgrade string indexing tests [#294](https://github.com/salesforce/TransmogrifAI/pull/294)
- Improved multi pick list map vectorizer test [#292](https://github.com/salesforce/TransmogrifAI/pull/292)
- Improvements of Vectorizer tests [#291](https://github.com/salesforce/TransmogrifAI/pull/291)
- Updated TextMapPivotVectorizerTest to use OpEstimatorSpec [#290](https://github.com/salesforce/TransmogrifAI/pull/290)
- Update TextTokenizerTest to use OpTransformerSpec [#289](https://github.com/salesforce/TransmogrifAI/pull/289)
- Add test for RealNNVectorizer [#288](https://github.com/salesforce/TransmogrifAI/pull/288)
- Improved OPCollectionHashingVectorizerTest test [#286](https://github.com/salesforce/TransmogrifAI/pull/286)
- Created new tests for OpCollection [#285](https://github.com/salesforce/TransmogrifAI/pull/286)
- Update names of transformer tests and files to match class names [#284](https://github.com/salesforce/TransmogrifAI/pull/286)
- Improved test by extending OpTransformerSpec [#283](https://github.com/salesforce/TransmogrifAI/pull/286)
- Skip writing empty stages & skip loading stages without uid-s [#282](https://github.com/salesforce/TransmogrifAI/pull/286)
- Skip serializing estimators + fix test + added empty data transform test [#281](https://github.com/salesforce/TransmogrifAI/pull/286)

Dependency updates:
N/A

## 0.5.2

Bug fixes:
- Fixed local scoring with multipicklist features [#243](https://github.com/salesforce/TransmogrifAI/pull/243)
- Fixed error messages in `DataCutter` and `DataBalancer` [#256](https://github.com/salesforce/TransmogrifAI/pull/256)
- Fixed bug in in model selector fit method [#251](https://github.com/salesforce/TransmogrifAI/pull/251)
- Fixed some Transmogrifier defaults to be modifiable / exposed [#232](https://github.com/salesforce/TransmogrifAI/pull/232)
- Fixed bug in `OpXGBoostClassificationModel` [#229](https://github.com/salesforce/TransmogrifAI/pull/229)
- Minor fixes / cleanup on notebooks, Helloworld examples, and developer guide [#226](https://github.com/salesforce/TransmogrifAI/pull/226), [#230](https://github.com/salesforce/TransmogrifAI/pull/230), [#240](https://github.com/salesforce/TransmogrifAI/pull/240), [#259](https://github.com/salesforce/TransmogrifAI/pull/259)

New features / updates:
- Added transformer classes for common math operations [#255](https://github.com/salesforce/TransmogrifAI/pull/255), [#257](https://github.com/salesforce/TransmogrifAI/pull/257)
- Added string transformers for substring search and valid email [#265](https://github.com/salesforce/TransmogrifAI/pull/265)
- Added scaler and descaler transformers [#223](https://github.com/salesforce/TransmogrifAI/pull/223)
- Added Raw Feature Filter results e.g., metrics, exclusion reasons to serialization and to `ModelInsights` [#237](https://github.com/salesforce/TransmogrifAI/pull/237), [#252](https://github.com/salesforce/TransmogrifAI/pull/252), [#258](https://github.com/salesforce/TransmogrifAI/pull/258), [#276](https://github.com/salesforce/TransmogrifAI/pull/276)
- Changed OpBinScoreEvaluator to allow for lift analysis [#233](https://github.com/salesforce/TransmogrifAI/pull/233)
- Added random param builder for random hyperparameter search in model selectors [#238](https://github.com/salesforce/TransmogrifAI/pull/238)
- Added possibility to return top K positives and top K negatives improvement for LOCO [#264](https://github.com/salesforce/TransmogrifAI/pull/264)
- Added a max cardinality percentage that can be set for pivot [#241](https://github.com/salesforce/TransmogrifAI/pull/241)
- Added minimum rows for scoring set in RawFeatureFilter [#250](https://github.com/salesforce/TransmogrifAI/pull/250)
- Allowed copying model instances across multiple threads [#270](https://github.com/salesforce/TransmogrifAI/pull/270)
- Added stub to allow loading models without workflow [#269](https://github.com/salesforce/TransmogrifAI/pull/269), [#272](https://github.com/salesforce/TransmogrifAI/pull/272)
- Made decision tree numeric bucketizer tests less flaky [#225](https://github.com/salesforce/TransmogrifAI/pull/225)
- Added Jupyter notebooks for samples [#231](https://github.com/salesforce/TransmogrifAI/pull/231)

Dependency updates:
- Switched to MLeap runtime from Aardpfark for local scoring [#249](https://github.com/salesforce/TransmogrifAI/pull/249), [#261](https://github.com/salesforce/TransmogrifAI/pull/261)

## 0.5.1

Bug fixes:
- Fix indices in LOCO for record-level insights and add more robust tests [#216](https://github.com/salesforce/TransmogrifAI/pull/216)
- Fix sorting in Prediction type for multiclass classification and add stronger tests [#213](https://github.com/salesforce/TransmogrifAI/pull/213)
- Fixing code generation bug with underscores in names [#208](https://github.com/salesforce/TransmogrifAI/pull/208)
- Correct some syntax/compilation errors in Titanic Binary Classification Docs Example [#202](https://github.com/salesforce/TransmogrifAI/pull/202)

New features / updates:
- Make some tests a little less flaky [#221](https://github.com/salesforce/TransmogrifAI/pull/221)
- Integrate helloworld project with Travis CI [#210](https://github.com/salesforce/TransmogrifAI/pull/210), [#212](https://github.com/salesforce/TransmogrifAI/pull/212)
- Use ParamGridBuilder in model selector grids to allow modifications [#206](https://github.com/salesforce/TransmogrifAI/pull/206)
- Use class.getName & update splitter meta parsing [#204](https://github.com/salesforce/TransmogrifAI/pull/204)
- Export model selector defaults + metadata fixes [#199](https://github.com/salesforce/TransmogrifAI/pull/199)
- Use OS specific path separator [#193](https://github.com/salesforce/TransmogrifAI/pull/193)
- Add transformer / estimator for text length calculation and options for using this as default behavior [#190](https://github.com/salesforce/TransmogrifAI/pull/190), [#195](https://github.com/salesforce/TransmogrifAI/pull/195)
- Allow conversion from Date and Timestamp Spark types to Date and DateTime TransmogrifAI types [#188](https://github.com/salesforce/TransmogrifAI/pull/188)

Dependency updates:
- Upgrade to Gradle 5.2 [#218](https://github.com/salesforce/TransmogrifAI/pull/218)
- Upgrade shadowjar plugin to 4.0.4 [#220](https://github.com/salesforce/TransmogrifAI/pull/220)

## 0.5.0

New features and bug fixes:

- XGBoost classification & regression models - EXPERIMENTAL [#44](https://github.com/salesforce/TransmogrifAI/pull/44)
- Add default param grid for xgboost [#175](https://github.com/salesforce/TransmogrifAI/pull/175)
- Fix ModelInsights for xgboost [#170](https://github.com/salesforce/TransmogrifAI/pull/170)
- Added Parquet reader [#169](https://github.com/salesforce/TransmogrifAI/pull/169)
- Added aggregate & conditional readers for Parquet [#172](https://github.com/salesforce/TransmogrifAI/pull/172)
- Evaluators check for empty data [#178](https://github.com/salesforce/TransmogrifAI/pull/178)
- Refactored splitter tests [#176](https://github.com/salesforce/TransmogrifAI/pull/176)
- Return scoring feature distributions from RawFeatureFilter [#171](https://github.com/salesforce/TransmogrifAI/pull/171)
- Using MapReduce Api for Avro Read Write [#150](https://github.com/salesforce/TransmogrifAI/pull/150)
- Improve test coverage for VectorsCombiner and make vector aggregator efficient [#168](https://github.com/salesforce/TransmogrifAI/pull/168)
- Time based aggregators [#167](https://github.com/salesforce/TransmogrifAI/pull/167)
- Ignore null values in meta + support floats [#166](https://github.com/salesforce/TransmogrifAI/pull/166)
- CLI command name fix + bump shadow plugin version + cleanup [#164](https://github.com/salesforce/TransmogrifAI/pull/164)
- Fix build.sbt example in readme [#165](https://github.com/salesforce/TransmogrifAI/pull/165)
- Removed an old test I added to check if Spark ran out of memory when calculating a correlation matrix (this is unnecessary and unhelpful) [#160](https://github.com/salesforce/TransmogrifAI/pull/160)
- Replace assert with require [#159](https://github.com/salesforce/TransmogrifAI/pull/159)
- Streaming histogram implementation [#152](https://github.com/salesforce/TransmogrifAI/pull/152)
- Added test and removed dead code for Sanity Checker dealing with map with same key [#153](https://github.com/salesforce/TransmogrifAI/pull/153)
- Fixed model insights exception when features are excluded from sanity checker correlation calculations [#147](https://github.com/salesforce/TransmogrifAI/pull/147)
- Added logging of response distribution to RFF [#146](https://github.com/salesforce/TransmogrifAI/pull/146)
- Use proper test ranges in feature converter test [#143](https://github.com/salesforce/TransmogrifAI/pull/143)
- Added support for DateType and TimestampType primitive spark types [#135](https://github.com/salesforce/TransmogrifAI/pull/135)
- Standardizing timezone to UTC [#138](https://github.com/salesforce/TransmogrifAI/pull/138)

Dependency upgrades & misc:
- XGBoost 0.81 [#180](https://github.com/salesforce/TransmogrifAI/pull/180)
- Spark 2.3.2 [#44](https://github.com/salesforce/TransmogrifAI/pull/44)
- Gradle 4.10.2 [#142](https://github.com/salesforce/TransmogrifAI/pull/142)
- Use OpenJDK8 for CircleCI builds + refactor build config [#140](https://github.com/salesforce/TransmogrifAI/pull/140)

## 0.4.0

New features and bug fixes:

- Allow to specify the formula to compute the text features bin size for `RawFeatureFilter` (see `RawFeatureFilter.textBinsFormula` argument) [#99](https://github.com/salesforce/TransmogrifAI/pull/99)
- Fixed metadata on `Geolocation` and `GeolocationMap` so that keep the name of the column in descriptorValue. [#100](https://github.com/salesforce/TransmogrifAI/pull/100)
- Local scoring (aka Sparkless) using Aardpfark. This enables loading and scoring models without Spark context but locally using Aardpfark (PFA for Spark) and Hadrian libraries instead. This allows orders of magnitude faster scoring times compared to Spark. [#41](https://github.com/salesforce/TransmogrifAI/pull/41)
- Add distributions calculated in `RawFeatureFilter` to `ModelInsights` [#103](https://github.com/salesforce/TransmogrifAI/pull/103)
- Added binary sequence transformer & estimator: `BinarySequenceTransformer` and `BinarySequenceEstimator` + plus the associated base traits [#84](https://github.com/salesforce/TransmogrifAI/pull/84)
- Added `StringIndexerHandleInvalid.Keep` option into `OpStringIndexer` (same as in underlying Spark estimator) [#93](https://github.com/salesforce/TransmogrifAI/pull/93)
- Allow numbers and underscores in feature names [#92](https://github.com/salesforce/TransmogrifAI/pull/92)
- Stable key order for map vectorizers [#88](https://github.com/salesforce/TransmogrifAI/pull/88)
- Keep raw feature distributions calculated in raw feature filter [#76](https://github.com/salesforce/TransmogrifAI/pull/76)
- Transmogrify to use smart text vectorizer for text types: `Text`, `TextArea`, `TextMap` and `TextAreaMap` [#63](https://github.com/salesforce/TransmogrifAI/pull/63)
- Transmogrify circular date representations for date feature types: `Date`, `DateTime`, `DateMap` and `DateTimeMap` [#100](https://github.com/salesforce/TransmogrifAI/pull/100)
- Improved test coverage for utils and other modules [#50](https://github.com/salesforce/TransmogrifAI/pull/50), [#53](https://github.com/salesforce/TransmogrifAI/pull/53), [#67](https://github.com/salesforce/TransmogrifAI/pull/67), [#69](https://github.com/salesforce/TransmogrifAI/pull/69), [#70](https://github.com/salesforce/TransmogrifAI/pull/70), [#71](https://github.com/salesforce/TransmogrifAI/pull/71), [#72](https://github.com/salesforce/TransmogrifAI/pull/72), [#73](https://github.com/salesforce/TransmogrifAI/pull/73)
- Match feature type map hierarchy with regular feature types [#49](https://github.com/salesforce/TransmogrifAI/pull/49)
- Redundant and deadlock-prone end listener removal [#52](https://github.com/salesforce/TransmogrifAI/pull/52)
- OS-neutral filesystem path creation [#51](https://github.com/salesforce/TransmogrifAI/pull/51)
- Make Feature class public instead hide it's ctor [#45](https://github.com/salesforce/TransmogrifAI/pull/45)
- Specify categorical variables in metadata [#120](https://github.com/salesforce/TransmogrifAI/pull/120)
- Fix fill geo location vectorizer values [#132](https://github.com/salesforce/TransmogrifAI/pull/132)
- Adding feature importance for new model types [#128](https://github.com/salesforce/TransmogrifAI/pull/128)
- Adding binaryclassification bin score evaluator [#119](https://github.com/salesforce/TransmogrifAI/pull/119)
- Apply DateToUnitCircleTransformer logic in raw feature filter transformations [130#](https://github.com/salesforce/TransmogrifAI/pull/130)

Breaking changes:
- Made case class to deal with model selector metadata [#39](https://github.com/salesforce/TransmogrifAI/pull/39)
- Made `FileOutputCommiter` a default and got rid of `DirectMapreduceOutputCommitter` and `DirectOutputCommitter` [#86](https://github.com/salesforce/TransmogrifAI/pull/86)
- Refactored `OpVectorColumnMetadata` to allow numeric column descriptors [#89](https://github.com/salesforce/TransmogrifAI/pull/89)
- Renaming `JaccardDistance` to `JaccardSimilarity` [#80](https://github.com/salesforce/TransmogrifAI/pull/80)
- New model selector interface [#55](https://github.com/salesforce/TransmogrifAI/pull/55). The breaking changes are related to return type and the way the parameters are passed into model selectors. Starting this version model selectors would return a single result feature of type `Prediction` (instead of a variable number of feature - `(pred, raw, prob)`). Example:
```scala
val (pred, raw, prob) = MultiClassificationModelSelector() // won't compile anymore
val prediction = MultiClassificationModelSelector() // ok!
```
Another change is the way parameters are passed into model selectors. Example:
```scala
BinaryClassificationModelSelector
  .withCrossValidation()
  .setLogisticRegressionRegParam(0.05, 0.1) // won't compile anymore
```
Instead one should do:
```scala
val lr = new OpLogisticRegression()
val models = Seq(lr -> new ParamGridBuilder().addGrid(lr.regParam, Array(0.05, 0.1)).build())
BinaryClassificationModelSelector
  .withCrossValidation(modelsAndParameters = models)
```
For more example on how to use new model selectors please refer to our documentation and helloworld examples.

Dependency upgrades & misc:
- CI/CD runtime improvements for CircleCI and TravisCI
- Updated Gradle to 4.10
- Updated `scala-graph` to `1.12.5`
- Updated `scalafmt` to `1.5.1`
- New `transmogrifai-local` subproject [#41](https://github.com/salesforce/TransmogrifAI/pull/41) introduces `aardpfark` and `hadrian` dependencies.


## 0.3.4
Performance improvements:
- Added featureLabelCorrOnly parameter in SanityChecker to only compute correlations between features and label (defaults to false)
- Added ignoreHashCorrelations parameter in SanityChecker that ignores correlations from hashed text features (defaults to false)
- Parallelize OP cross validation and set default validation parallelism to 8
- Added warmup in concurrent checks

New features and bug fixes:
- Replace deprecated 'forceSharedHashSpace' param with HashingStrategy
- Added explicit annotations for all classes with generic collections that use JsonUtils
- Added .transmogrify shortcut for arrays of features
- Removed referencing UID from a case object
- DecisionTree & DropIndices stages tests now use the OP spec base classes
- Added map features removed by RFF to model insights
- Pretty print model summaries
- Ensure OP Models are portable across environments
- Ignore _ in simple streaming avro file reader
- Updated evaluators so they can work with either Prediction type feature or three input features 
- Added Algebird kryo registrar
- Make Sure that SmartTextVectorizerModel can be serialized to/from json

Dependency upgrades:
- Upgraded to Scala 2.11.12
- Updated Gradle to 4.9 & bump Scalastyle plugin to 1.0.1


## Archived Releases
## 3.3.3

1. Convert some more stages tests to use OP stages specs (#241)
2. Changed error to occur only when all labels are removed (#237)
3. Fixes for writing/reading stages in OpPipelineStageSpec tests (#235)
4. Add files via upload (#233)
    workflow description figures
5. Stop words changes to text analyzers and bug fixes (#230)
6. Update README.md (#229)
7. Remove null leakage checks for text features from sanity checker (#228)
    Update sanity checker shortcut with protectTextSharedHash param (#234)
8. Remove JSD check for date + datetime features in RFF (#227)
9. Introduced FeatureBuilder.fromDataFrame function allowing materializing features from a DataFrame (#226)
10. Get rid of ClassTags in OP models (#225)
11. Test if transformer transforms the data correctly after being loaded (#223)
12. Update to BSD-3 license (#218)
    Some more licenses (#221)
13. Changed from extending to wrapping spark models.
    wrapped spark model classed using reflection (part 1 of 2) (#216)
    wrapped spark estimators so that they return op wrapped models with prediction return type (part 2a of 2) (#222)
    wrapped spark estimators for new models added (part 2b of 2) (#238)
    Moved code out of spark ml workspace and added comments - no code changes after tickets (#239)
14. Change ootb transformers to use OPTransformerSpec for tests (#215)
15. Move base stages to features sub project + test classes and specs (#214)
16. Better clues when asserting stages (#213)
17. Implement multi-class threshold metrics (#212)
18. NameEntityRecognizer (NER) transformer (#209)
19. Allow customizing feature type equality in op test transformer/estimator specs (#207)
20. Threshold metrics bug fix (#204)
    use prediction rather than raw prediction
21. Added an extra OpEstimatorBaseSpec base class with loosen model type boundaries to allow testing Spark wrapped estimators (#203)
    Fix package access level on OpEstimatorBaseSpec (#205)
    internal OP test base class 
22. Fast materializer method FeatureTypeSparkConverter by full feature type name (#202)
23. Added UID.reset() before tests so that all workflows will generate the same feature names (#201)
24. Added add/subtract operations for Spark ml Vector types (#200)
25. workflow cleanup (#199)
26. Fix TextMapNullEstimator to count a null when text entirely removed by tokenizer (#198)
    fix the issue that certain text strings can be entirely removed by our tokenizers, but  null tracking step for text map vectorizers just checks for the presence of a key 
27. Workflow CV Fixes (#196)
    fix dead lock in OpCrossValidation.findBestModel happened due to the fact that when running splits processing in parallel these threads would try to access spake stage params on the same stages.
28. Update ternary, quaternary and sequence transformer/estimator bases tests (#195)
29. Enabling null-label leakage detection in RawFeatureFilter (#191, #192, #193)
30. Feature Type values docs (#190)
31. Bump up lucene version and add lucene-opennlp package (#188)
32. Minor README cleanup (#187, #189)
33. Test specs for OP stages (#186)
34. Adding pr_curve, roc_curve metrics (#184)
35. Create hash space strategy param (#182)
36. Make new Cross Validation (#181)
37. Avoid reseting UID in every test, but only do it when necessary (#180)
38. Upgrade to gradle 4.7 (#179)
39. Added OpTransformer.transformKeyValue to allow transforming Map and any other key/value types (#178) in preparation for sparkless scoring
40. Adding autoBucketize to transmogrify for numerics & numeric maps + pass in optional label #159 
41. Autobucketizing for numeric maps should not fail if map is empty, instead we generate empty column for empty numeric map #231 

Migration guide:
1. OpLogisticRegression() is in progress (evaluator needs updates)
    may use BinaryClassificationModelSelector() instead
2. Need to add .setProbabilityCol($probCol) to evaluator in workflow definition to make sure that the evaluator will get the correct probability column to do the calculation


## 3.3.1
- SanityChecker performance improvements
- Introduced `FeatureBuilder.fromDataFrame` function allowing materializing features from a DataFrame. Example usage:
```scala
case class MyClass(s: String, l: Long, label: Double)
val data: DataFrame = Seq(MyClass("blah1", 10, label = 2.0)).toDS.toDF()
val (label, Array(fs, fl)) = FeatureBuilder.fromDataFrame[RealNN](data, response = "label")
```
- Helloworld: added a minimalistic classifier based on the Titanic dataset `com.salesforce.hw.titanic.OpTitanicMini`

## 3.3.0
1. Json4s extension to serialize Joda time arguments with op stages
2. Correctly produce json for OpWorkflowRunnerConfig
3. Added SmartTextVectorizer and SmartTextMapVectorizer
4. Update OP type hierarchy image
5. Name update bug fix in `SwThreeStageBinaryEstimator`
6. Added feature type conversion shortcuts for floats
7. Smart bucketizer for numeric map values based on a Decision Tree classifier
8. Allow serializing HashAlgorithm enum in a stage argument
9. Update Model Insights to have features excluded by raw feature filter
10. Redesign DataCutter for new Cross-Validation/Train-Validation-Split
11. Allow setting log level in Sanity Checker
12. Move reference data out of OPMapVectorizerModelArgs
13. Made maps params needed for feature parity in builder + increase defaultNumOfFeatures and maxNumOfFeatures for hashing
14. Association rule confidence/support checks
    1. Added maxConfidences function to OpStatistics that calculates the max confidence per row of the contingency matrix, along with the support of that row. Refactored SanityCheckerSummary metadata so that everything coming from the same feature group (contingency matrix) are grouped together.
15. anity checker summary metadata redesign
16. Tests indicator group collapsing for sanity checker
17. Added loco record insights
18. Redesign DataBalancer
    1. (Internal optimization) With new Cross Validation, the same DataBalancer (i.e. with the same fractions) will run many times. Estimation is not necessary, hence no need to count over and over again.
19. Model selector modified in order to have cross validation and train-split validation called on it rather than running them internally.
20. Added ability to set output name on all stages
21. Allow suppressing arg parse errors
22. Modify workflow to run cv on all stages with label mixed in
    1. **New Restriction:** OpWorkflows can only contain at most 1 Model Selector, an error will be thrown otherwise.
23. Updated default for binary model selector evaluator
    1. **New Default:** Area under PR curve is default value.
24. Added FeatureBuilder.fromRow and FeatureLike.asRaw methods.
25. Added constructor parameter `stratify` in cross-validation and train-validation split for stratification.
26. Added ability to use raw feature filter to workflow.
27. Added `RawFeatureFilter` class
28. Extend Cramer's V to work with MultiPickLists
    1. Added calculation of Cramer's V on MultiPickList fields, computed from the max of all the 2x2 Cramer's V values on each individual choice of the MultiPickList. Updated methods in OpStatistics to return chi squared statistic and p value.
29. Added Prediction feature type
    1. `Prediction` is a new `NonNullable` feature type that inherits from `RealMap`. It requires at least a `prediction: Double` to be provided, otherwise the error is thrown at construction.
    2. `Prediction` can also contain the `rawPrediction: Array[Double]` and `probability: Array[Double]` values.
30. Added UID.reset and UID.count + tests
31. Modify OpParams to provide read locations for two readers in Assessor (RawFeatureFilter) stage
32. Error on null/empty in `RealNN` + make `OPVector` nullable
    1. `RealNN` now throws an exception on null/empty values at construction
    2. `OPVector` is now a nullable type
    3. Removed `OpNumeric.``map` and `OpNumber.toDouble(default)`
33. Make param settings take priority over code settings and allow setting params that do not correspond to an underlying spark param
34. Drop indices transformer
35. Bug fix in calculating max sibling correlation
36. Added Date To Unit Circle Transformer
    1. Implements a transformer of a Date or DateTime field into a cartesian coordinate representation of an extracted time period on the unit circle.

Migration guide:
1. Use `com.salesforce.op.utils.json.EnumEntrySerializer.json4s` instead of `EnumEntrySerializer.apply` for creating JSON4S formats.
2. Make sure to specify `OpParams.alternateReaderParams` when using main constructor (can default to `Map.empty`).
3. `RealNN` now can only be created from an actual Double/Long/Int value or with a default value/behavior provided:
```scala
RealNN(0.0) // ok
1.0.toRealNN // ok
Real(None).value.toRealNN(-1.0) // ok, but default value is a requirement now
Real(None).value.toRealNN(throw new RuntTimeException("RealNN cannot be empty")) // ok
Real(None).value.toRealNN // NOT ok
```
4. OP pipeline stage `operationName` was renames to `getOperationName`


## 3.2.4
- OpVectorColumnMetadata.hasParentOfType should be using exists
- Made Splitter methods public and not final so can extend classes
- Date time map now has reference date
- Make TestFeatureBuilder correctly generate originStage
- Allow specifying cutOff logic for conditional readers
- Moved stage fit into trait that can be reused in CV and TS 
- Use array when applying op transformers
- Change `minInfoGain` Param default in `DecisionTreeBucketizer` 
- Aggregator fixes
- Fixed max modeling default
- Call stage.setInputSchema before transformSchema is called
- Delete the cli test dir recursively
- Refactored op read/write stage tests￼
- Bumped the minimum sample size up to 1000 for SanityChecker
- Added internal ValidatedModel type for to hold intermediate in ModelSelector
- Flattened cross validation classes so can make workflow level CV 
- Upgrade to gradle 4.5.1 + fix date time utils tz
- Added persist in sanity checker after sample
- Fixed catalyst issue when training very wide datasets
- Added MaxRealNN, MinRealNN, MeanRealNN + cleanup some aggregator tests
- CLI fixes
- Allow disable version props generation
- Add Scalastyle check for license header 
- Strip HTML tags for text features

## 3.2.3
- Upgraded to Spark 2.2.1
- Numeric bucketizers fixes: including metadata and integration with sanity checker.
- Introduced split inclusion parameter for numerical bucketizers controlling should the splits be left or right inclusive (`numeric.bucketize(splitInclusion)` and `numeric.autoBucketize(splitInclusion)`).
- Introduced an option for numerical bucketizers to allow tracking invalid values such as NaN, -Inf, Inf or values that fall outside the buckets. (`numeric.bucketize(trackInvalid)` and `numeric.autoBucketize(trackInvalid)`).
- Finalized the unification of vectorization for ALL the OP types, including `Text`, `TextArea`, `Base64`, `Phone`, `URL` and `Geolocation`.
- Fixed track null behavior of vectorization for `MulitPicklistMap` to match `MultiPicklist`
- OP cli `--auto` properly identifies the input csv schema as expected
- OP cli generated code is now prettier
- Minor bug fixes in vectorizers metadata and ctor args for sanity checker model
- Added null tracking for map vectorizers
- Added null tracking for hashed text features
- Sanity Checker: fixed issue in feature removal for sibling features when one has a correlation of NaN with the label
- Fixes for Decision Tree bucketizer vector metadata to allow it working with Sanity Checker
- Correlation based record level insights
- All model classes are now made public with `private[op]` ctors
- Added inclusion indicators in bucket labels
- Workflow now creates holdout during fitting after raw data creation and applies eval
- Simplified OP CLI generated template
- Added `VersionInfo` - which allows access to project version and git info at runtime and include it in `AppMetrics`
- Make most `OpWorkflowRunner` ctor params optional and deprecate old ctors
- `OpWorkflowRunner.run` now requires spark `StreamingContext` to be present
- Implemented a new run type allowing streaming score (`--run-type=streamingScore`)

## 3.2.0
- Implemented Model Insights (acessible through `model.modelInsights(feature)`)
- Better hyperpameter defaults for model selectors
- Implemented smart binning estimator for continuous variables based on a decision tree classifier (accessible through `numeric.autoBucketize(label, trackNulls)`, while regular bucketizer is available as `numeric.bucketize(trackNulls, splits, bucketLabels)`)
- Added data cutter for multiclass model selectors - creates a data splitter that will split data into training and test set filtering out any labels that don't
meet the minimum fraction cutoff or fall in the top N labels specified.
- Base64 features vectorization - that uses `detectMimeTypes` underneath, then converts the results into `PickList` and vectorizes it
- Unified vectorization behaviour between numerical maps and numerical base feature types
- Unified vectorization behaviour between text maps and text base feature types that are pivoted
- Added an option to sanity checker to remove all features descended from a parent feature (parent is direct parent before vectorization) which has derived features that meet the exclusion criteria (`sanityChecker.setRemoveFeatureGroup`)
- Renamed sampleLimit to sampleUpperLimit and added sampleLowerLimit options for sanity checker
- Collect and expose spark stage metrics - `OpSparkListener` now allows to collect and expose stage metrics and it's controlled using `OpParams.collectStageMetrics` parameter (`false` by default). The metrics are available through `OpWorkflowRunner.addApplicationEndHandler`.
- OP CLI gen: allows providing answers file, auto detecting schemas and autogeneration of avro schema from data.

Minor fixes:
- Add null tracking to `NumericBucketizer`
- Add sequence aggregator for calculating means by key of RealMaps
- Added commutative group aggregator - It is good for sliding window aggregation - can subtract expired data. As an example, ExtendedMultiset - it counts words, and can subtract. So, it also can have negative counters, for borrowed words or something.
- Fixing random phones generator: US phone numbers are now correctly generated

Migration Guide:
- `TextMapVectorizer` has been renamed to `TextMapPivotVectorizer` to make its behavior more apparent. A new `TextMapHashingVectorizer` will be released in a future version.


