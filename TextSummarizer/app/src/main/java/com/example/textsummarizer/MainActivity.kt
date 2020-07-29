package com.example.textsummarizer

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
class MainActivity : AppCompatActivity() {
    private var summaryValue: TextView? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val predict: Button =findViewById(R.id.button)
        predict.setOnClickListener {
            val feedValue: TextView=findViewById(R.id.editTextTextMultiLine)
            summaryValue=findViewById(R.id.textView3)
            //val textSummary= MainActivity.TextSummarizer(feedValue.text.toString())
            Textsummarize(feedValue.text.toString())

        }




    }
    fun Textsummarize(textReqVal:String) {

        val retrofit = Retrofit.Builder().baseUrl(BaseUrl)
            .addConverterFactory(GsonConverterFactory.create()).build()
        val service = retrofit.create(Summary::class.java)
        val call = service.GetSummary(textReqVal)
        Log.i("TAG","call "+call)
        call.enqueue(object : Callback<SummaryResponse> {
            override fun onResponse(
                call: Call<SummaryResponse>,response: Response<SummaryResponse>) {
                if (response != null) {
                    if (response.code() == 200) {
                        val summaryResponseValue = response.body()

                        summaryValue!!.text =summaryResponseValue.summary.toString()


                    }
                }
            }

            override fun onFailure(call: Call<SummaryResponse>, t: Throwable) {
                summaryValue!!.text = t.message.toString()

            }
        })


    }
    companion object{

        var BaseUrl = "https://f4a5ade44f00.ngrok.io"


    }
}