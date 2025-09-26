 # Flesh Color Filter

<p>
<img width="600" height="538" alt="image" src="https://github.com/user-attachments/assets/2381fba9-30c7-4c09-a1db-ebb466f8e759" />
</p>

# 데이터 처리

| 항목     | 내용                                                                                                 |
|:-----:|------------------------------------------------------------------------------------------------------|
| **입력** | `den`, `r_in` / `g_in` / `b_in` (각 4비트)                                                           |
| **처리** | 4b → 8b Nibble 복제 → 정수형 YCbCr 근사 변환 → Cb/Cr 범위 + (r > g, b) 조건으로 살색 여부 판정             |
| **출력** | 살색이면 `r_out`, `g_out`, `b_out` = `4'hF` (흰색), 아니면 `4'h0` (검정)                             |

# Code
```verilog
`timescale 1ns/1ps

module flesh_color (
    input  logic       den,     
    input  logic [3:0] r_in,
    input  logic [3:0] g_in,
    input  logic [3:0] b_in,

    output logic [3:0] r_out,
    output logic [3:0] g_out,
    output logic [3:0] b_out
);

    wire [7:0] R8 = {r_in, r_in};
    wire [7:0] G8 = {g_in, g_in};
    wire [7:0] B8 = {b_in, b_in};

    // Cb = 128 + (-43R -85G +128B)/256
    // Cr = 128 + ( 128R -107G -21B)/256
    wire signed [17:0] cb_acc = -18'sd43 * $signed({1'b0,R8})
                              + -18'sd85 * $signed({1'b0,G8})
                              +  18'sd128* $signed({1'b0,B8});
    wire signed [17:0] cr_acc =  18'sd128* $signed({1'b0,R8})
                              + -18'sd107* $signed({1'b0,G8})
                              + -18'sd21 * $signed({1'b0,B8});
    wire [7:0] Cb = 8'd128 + (cb_acc >>> 8);  
    wire [7:0] Cr = 8'd128 + (cr_acc >>> 8);

    // 기본: 77 ≤ Cb ≤ 127, 133 ≤ Cr ≤ 173
    localparam [7:0] CB_MIN = 8'd77,  CB_MAX = 8'd127;
    localparam [7:0] CR_MIN = 8'd133, CR_MAX = 8'd173;
    wire is_skin = den
                && (Cb >= CB_MIN) && (Cb <= CB_MAX)
                && (Cr >= CR_MIN) && (Cr <= CR_MAX)
                && (r_in > g_in) && (r_in > b_in);

    assign {r_out, g_out, b_out} = is_skin ? 12'hFFF : 12'h000;

endmodule
```







