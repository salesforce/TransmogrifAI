/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
