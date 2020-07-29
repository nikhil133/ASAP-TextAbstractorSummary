package com.example.textsummarizer

import retrofit2.Call
import retrofit2.http.GET
import retrofit2.http.Query

interface Summary {
    @GET("/app/v1/incoming")
    fun GetSummary(@Query("text")text: String): Call<SummaryResponse>
}