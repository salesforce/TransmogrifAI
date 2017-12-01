/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

// http://www.princeton.edu/~otorres/Regression101R.pdf
// education. Average education of occupationalincumbents, years, in 1971.
// income. Average income of incumbents, dollars, in 1971.
// women. Percentage of incumbents who are women.
// prestige. Pineo-Porter prestige score for occupation, from a social survey conducted in the mid-1960s.
case class Prestige(education: Double, income: Double, women: Double, prestige: Double)

trait PrestigeData {

  val prestigeSeq = Seq(
    Prestige(13.11, 12351, 11.16, 68.8),
    Prestige(12.26, 25879, 4.02, 69.1),
    Prestige(12.77, 9271, 15.70, 63.4),
    Prestige(11.42, 8865, 9.11, 56.8),
    Prestige(14.62, 8403, 11.68, 73.5),
    Prestige(15.64, 11030, 5.13, 77.6),
    Prestige(15.09, 8258, 25.65, 72.6),
    Prestige(15.44, 14163, 2.69, 78.1),
    Prestige(14.52, 11377, 1.03, 73.1),
    Prestige(14.64, 11023, 0.94, 68.8)
  )

}
