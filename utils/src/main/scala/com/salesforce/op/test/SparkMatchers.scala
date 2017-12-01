/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import language.postfixOps
import org.apache.spark.sql.DataFrame
import org.scalatest.Assertion
import org.scalatest.Matchers._

/**
 * Convenient matchers for spark functionality.
 *
 * Examples of usage:
 * {{{
 *   in(myDF) allOf "MyNumericColumn" should beBetween(-1, 1)
 *
 *   in(myDF) someOf "MyStringColumn" should (
 *     (x: String) => (x contains "Country") || (x contains "State")
 *  )
 *
 *  in(myDF) noneOf "UserName" shouldContain "Snowden"
 * }}}
 */
object SparkMatchers {

  /**
   * Checks if a value is between boundaries
   * @param from lower boundary
   * @param to upper boundary
   * @param x the value we check
   * @return true iff it is between
   *
   * Examples of usage:
   * {{{
   *   in(myDF) allOf "Paychecks" should beBetween(5000.00, 500000.00)
   * }}}
   */
  def beBetween(from: Double, to: Double)(x: Double): Boolean = x >= from && x <= to

  sealed class ColumnMatcher private[test]
  (df: DataFrame, columnName: String, expected: Long => Boolean, inv: Boolean => Boolean = identity)
    extends Serializable {
    def should[T](predicate: T => Boolean): Assertion = {
      val count = df.filter{
        row => inv(predicate(row.getAs[T](columnName)))
      }.count

      expected(count) shouldBe true
    }
    def shouldContain(what: String): Assertion = should[String]((s: String) => s contains what)
    def shouldBe[T](sample: T): Assertion = should[T](sample == _)

// The following trick, that would allow stuff like { in(ds) someOf "f2" should be > 1.5 },
// won't work in distributed environment: scalatest.BeWord is final and not serializable
//    def should[T](matcher: Matcher[T] with Serializable): Assertion = {
//      val predicate = (t: T) => matcher.apply(t).matches
//      should[T](predicate)
//    }

  }

  private[test] sealed class FrameChecker(df: DataFrame) extends Serializable {

    // number of rows satisfying the predicate expected to be >0
    def someOf(columnName: String): ColumnMatcher = new ColumnMatcher(df, columnName, 0 <)

    // number of rows satisfying the predicate expected to be 0
    def noneOf(columnName: String): ColumnMatcher = new ColumnMatcher(df, columnName, 0==)

    // number of rows not satisfying the predicate expected to be 0
    def allOf (columnName: String): ColumnMatcher = new ColumnMatcher(df, columnName, 0==, !_)
  }

  /**
   * Instantiates a frame checker on given data frame
   * @param df the data frame
   * @return the frame checker
   */
  def in(df: DataFrame): FrameChecker = new FrameChecker(df)
}
