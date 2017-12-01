/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import org.scalatest.Matchers
import org.scalatest.prop.{PropertyChecks, TableFor1}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}


trait ConcurrentCheck {
  self: PropertyChecks with Matchers =>

  def forAllConcurrentCheck[T](
    numThreads: Int = 10,
    numInstancesPerThread: Int = 50000,
    atMost: Duration = 10.seconds,
    table: TableFor1[T],
    functionCheck: T => Unit
  ): Unit = {

    val started = System.currentTimeMillis()
    forAll(table) { ft =>
      def testOne() = {
        var i = 0
        while (i < numInstancesPerThread) {
          functionCheck(ft)
          i += 1
        }
      }

      val all = Future.sequence((0 until numThreads).map(_ => Future(testOne())))
      val res = Await.result(all, atMost)
      res.length shouldBe numThreads
    }
    val elapsed = System.currentTimeMillis() - started
    println(
      s"Tested with $numThreads concurrent threads. Elapsed: ${elapsed}ms. " +
        s"Created ${numInstancesPerThread * numThreads} instances with average " +
        (System.currentTimeMillis() - started) / (numInstancesPerThread * numThreads).toDouble +
        "ms per instance creation."
    )
    elapsed should be <= atMost.toMillis
  }

}
