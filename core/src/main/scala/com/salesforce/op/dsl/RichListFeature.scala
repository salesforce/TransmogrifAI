/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{DateList, DateTimeList, Geolocation, OPVector, TextList}
import com.salesforce.op.stages.impl.feature._
import com.salesforce.op.stages.sparkwrappers.specific.OpTransformerWrapper
import org.apache.spark.ml.feature.{HashingTF, NGram, StopWordsRemover}
import org.joda.time.DateTime


trait RichListFeature {

  self: RichFeature with RichVectorFeature =>

  /**
   * Enrichment functions for TextList Feature
   *
   * @param f TextList Feature
   */
  implicit class RichTextListFeature(val f: FeatureLike[TextList]) {

    /**
     * Apply hashing term frequency transformation.
     *
     * @param numTerms number of features (> 0)
     * @param binary      if true, all non zero counts are set to 1.0
     */
    def tf(
      numTerms: Int = TransmogrifierDefaults.DefaultNumOfFeatures,
      binary: Boolean = TransmogrifierDefaults.BinaryFreq
    ): FeatureLike[OPVector] = {
      val htf = new HashingTF().setNumFeatures(numTerms).setBinary(binary)
      val tr = new OpTransformerWrapper[TextList, OPVector, HashingTF](htf, UID[HashingTF])
      f.transformWith(tr)
    }

    /**
     * Apply Term frequency-inverse document frequency (TF-IDF), a feature vectorization method to reflect the
     * importance of a term to a document in the corpus
     *
     * @param numTerms number of features (> 0)
     * @param binary      if true, all non zero counts are set to 1.0
     * @param minDocFreq  minimum number of documents in which a term should appear for filtering
     */
    def tfidf(
      numTerms: Int = TransmogrifierDefaults.DefaultNumOfFeatures,
      binary: Boolean = TransmogrifierDefaults.BinaryFreq,
      minDocFreq: Int = 0
    ): FeatureLike[OPVector] =
      f.tf(numTerms = numTerms, binary = binary).idf(minDocFreq = minDocFreq)

    /**
     * Convert array of strings into one vector using word2vec method
     *
     * @param maxIter           Number of iterations
     * @param maxSentenceLength Maximum length (in words) of each sentence in the input data
     * @param minCount          Minimum number of times a token must appear to be included in the word2vec model's
     * @param numPartition      Number of partitions for sentences of words
     * @param stepSize          Initial learning rate
     * @param windowSize        Window size (context words from [-window, window])
     * @param vectorSize        Dimension of the code that you want to transform from words
     * @return
     */
    def word2vec(
      maxIter: Int = 1,
      maxSentenceLength: Int = 1000,
      minCount: Int = 5,
      numPartition: Int = 1,
      stepSize: Double = 0.025,
      windowSize: Int = 5,
      vectorSize: Int = 100
    ): FeatureLike[OPVector] = {
      val w2v = new OpWord2Vec()
        .setMaxIter(maxIter)
        .setMaxSentenceLength(maxSentenceLength)
        .setMinCount(minCount)
        .setNumPartitions(numPartition)
        .setStepSize(stepSize)
        .setWindowSize(windowSize)
        .setVectorSize(vectorSize)
      f.transformWith(w2v)
    }

    /**
     * Converts array of strings into a count vector
     *
     * @param binary    Binary toggle to control the output vector values. If True, all nonzero counts are set to 1.
     * @param minDF     Minimum number of documents a term must appear in to be included in the vocabulary.
     *                  If this is an integer greater than or equal to 1, this specifies the number of documents
     *                  the term must appear in.
     *                  if this is a double in [0,1), then this specifies the fraction of documents.
     * @param minTF     Minimum number of times a term must appear in a document.
     *                  If this is an integer greater than or equal to 1, then this specifies a count.
     *                  If this is a double in [0,1), then this specifies a fraction.
     * @param vocabSize Max size of the vocabulary.
     *                  CountVectorizer will build a vocabulary that only considers the top vocabSize terms ordered
     *                  by term frequency across the corpus.
     * @return
     */
    def countVec
    (
      binary: Boolean = false,
      minDF: Double = 1.0,
      minTF: Double = 1.0,
      vocabSize: Int = 1 << 18
    ): FeatureLike[OPVector] = {
      val cntVec = new OpCountVectorizer().setBinary(binary).setMinDF(minDF).setMinTF(minTF).setVocabSize(vocabSize)
      f.transformWith(cntVec)
    }

    /**
     * A feature transformer that converts the input array of strings into an array of n-grams.
     * Note: Null values in the input list are ignored.
     * It returns a list of n-grams where each n-gram is represented by a space-separated string of words.
     *
     * When the input is empty, an empty array is returned.
     * When the input array length is less than n (number of elements per n-gram), no n-grams are returned.
     *
     * @param n number elements per n-gram (>=1)
     * @return
     */
    def ngram(n: Int = 2): FeatureLike[TextList] = {
      val ngrm = new NGram().setN(n)
      val tr = new OpTransformerWrapper[TextList, TextList, NGram](ngrm, UID[NGram])
      f.transformWith(tr)
    }

    /**
     * A feature transformer that filters out stop words from input.
     * Note: null values from input array are preserved unless adding null to stopWords explicitly.
     *
     * @param stopWords     the words to be filtered out (default: English stop words)
     *                      See [[StopWordsRemover.loadDefaultStopWords()]] for all supported languages.
     * @param caseSensitive whether to do a case-sensitive comparison over the stop words
     * @return
     */
    def removeStopWords(
      stopWords: Array[String] = StopWordsRemover.loadDefaultStopWords("english"),
      caseSensitive: Boolean = false
    ): FeatureLike[TextList] = {
      val remover = new StopWordsRemover().setStopWords(stopWords).setCaseSensitive(caseSensitive)
      val tr = new OpTransformerWrapper[TextList, TextList, StopWordsRemover](remover, UID[StopWordsRemover])
      f.transformWith(tr)
    }


    /**
     * Apply Term frequency-inverse document frequency (TF-IDF), a feature vectorization method to reflect the
     * importance of a term to a document in the corpus. Results in a vector representation of text
     *
     * @param numTerms number of features (> 0)
     * @param binary      if true, all non zero counts are set to 1.0
     * @param minDocFreq  minimum number of documents in which a term should appear for filtering
     */
    def vectorize(
      numTerms: Int,
      binary: Boolean,
      minDocFreq: Int,
      others: Array[FeatureLike[TextList]] = Array.empty
    ): FeatureLike[OPVector] = {
      val vectors: Seq[FeatureLike[OPVector]] = (f +: others).toSeq.map(_.tfidf(numTerms, binary, minDocFreq))
      vectors.combine()
    }

  }

  /**
   * Enrichment functions for DateList Feature
   *
   * @param f DateList Feature
   */
  implicit class RichDateListFeature(val f: FeatureLike[DateList]) {

    /**
     * Apply DateList vectorizer: Converts a sequence of DateLists features into a vector feature.
     *
     * DateListPivot can specify:
     * 1) SinceFirst - replace the feature by the number of days between the first event and reference date
     * 2) SinceLast - replace the feature by the number of days between the last event and reference date
     * 3) ModeDay - replace the feature by a pivot that indicates the mode of the day of the week
     * Example : If the mode is Monday then it will return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
     * 4) ModeMonth - replace the feature by a pivot that indicates the mode of the month
     * 5) ModeHour - replace the feature by a pivot that indicates the mode of the hour of the day.
     *
     * @param others        other features of same type
     * @param dateListPivot name of the pivot type from [[DateListPivot]] enum
     * @param referenceDate reference date to compare against when [[DateListPivot]] is [[SinceFirst]] or [[SinceLast]]
     * @param trackNulls    option to keep track of values that were missing
     * @return result feature of type Vector
     */
    def vectorize
    (
      dateListPivot: DateListPivot,
      referenceDate: DateTime = TransmogrifierDefaults.ReferenceDate,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      others: Array[FeatureLike[DateList]] = Array.empty
    ): FeatureLike[OPVector] = {
      new DateListVectorizer()
        .setInput(f +: others)
        .setPivot(dateListPivot)
        .setReferenceDate(referenceDate)
        .setTrackNulls(trackNulls)
        .getOutput()
    }

  }

  /**
   * Enrichment functions for DateList Feature
   *
   * @param f DateList Feature
   */
  implicit class RichDateTimeListFeature(val f: FeatureLike[DateTimeList]) {

    /**
     * Apply DateList vectorizer: Converts a sequence of DateTimeLists features into a vector feature.
     *
     * DateListPivot can specify:
     * 1) SinceFirst - replace the feature by the number of days between the first event and reference date
     * 2) SinceLast - replace the feature by the number of days between the last event and reference date
     * 3) ModeDay - replace the feature by a pivot that indicates the mode of the day of the week
     * Example : If the mode is Monday then it will return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
     * 4) ModeMonth - replace the feature by a pivot that indicates the mode of the month
     * 5) ModeHour - replace the feature by a pivot that indicates the mode of the hour of the day.
     *
     * @param others        other features of same type
     * @param dateListPivot name of the pivot type from [[DateListPivot]] enum
     * @param referenceDate reference date to compare against when [[DateListPivot]] is [[SinceFirst]] or [[SinceLast]]
     * @param trackNulls    option to keep track of values that were missing
     * @return result feature of type Vector
     */
    def vectorize
    (
      dateListPivot: DateListPivot,
      referenceDate: DateTime = TransmogrifierDefaults.ReferenceDate,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      others: Array[FeatureLike[DateTimeList]] = Array.empty
    ): FeatureLike[OPVector] = {
      new DateListVectorizer()
        .setInput(f +: others)
        .setPivot(dateListPivot)
        .setReferenceDate(referenceDate)
        .setTrackNulls(trackNulls)
        .getOutput()
    }

  }

  /**
   * Enrichment functions for Geolocation Feature
   *
   * @param f DateList Feature
   */
  implicit class RichGeolocationFeature(val f: FeatureLike[Geolocation]) {

    /**
     * Apply Geolocation vectorizer: Converts a sequence of Geolocation features into a vector feature.
     *
     * @param others       other features of same type
     * @param fillValue    value to pull in place of nulls
     * @param trackNulls   keep tract of when nulls occur in a separate column that will be added to the output
     * @param fillWithMean replace missing values with mean (as apposed to constant provided in fillValue)
     * @return
     */
    def vectorize
    (
      fillWithMean: Boolean,
      trackNulls: Boolean,
      fillValue: Geolocation = TransmogrifierDefaults.DefaultGeolocation,
      others: Array[FeatureLike[Geolocation]] = Array.empty
    ): FeatureLike[OPVector] = {
      val stage = new GeolocationVectorizer()
        .setInput(f +: others)
        .setTrackNulls(trackNulls)
      if (fillWithMean) stage.setFillWithMean() else stage.setFillWithConstant(fillValue)
      stage.getOutput()
    }

  }

}
