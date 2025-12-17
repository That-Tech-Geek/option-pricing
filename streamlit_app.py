import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

# ==========================================
# 1. HELPER FUNCTIONS (Black-Scholes)
# ==========================================
def bs_call_price(S, K, T, r, sigma):
    """Standard Black-Scholes Call Price"""
    if T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    """Standard Black-Scholes Put Price"""
    call = bs_call_price(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)

# ==========================================
# 2. SABR MODEL (Hagan 2002)
# ==========================================
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan 2002 SABR Implied Volatility approximation.
    """
    # Handle ATM case to avoid division by zero
    if abs(F - K) < 1e-5:
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + (((1 - beta)**2 / 24) * alpha**2 / (F**(2 - 2*beta)) + 
                     (rho * beta * nu * alpha) / (4 * F**(1 - beta)) + 
                     ((2 - 3 * rho**2) / 24) * nu**2) * T
        return term1 * term2

    # Standard case
    log_FK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * log_FK
    
    # x(z) function
    # Safety check for z close to 1 with rho close to 1
    if abs(z - rho) < 1e-7: 
        x_z = 1.0 # Fallback approximation
    else:
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    numerator = alpha * (1 + (((1 - beta)**2 / 24) * log_FK**2 + ((1 - beta)**4 / 1920) * log_FK**4))
    
    denominator = FK_beta * (1 + ((1 - beta)**2 / 24) * log_FK**2 + ((1 - beta)**4 / 1920) * log_FK**4) # Simplified denom for display consistency usually
    # Correct Hagan Denominator adjustment:
    denominator = FK_beta * (z / x_z) 

    brackets = 1 + (((1 - beta)**2 / 24) * alpha**2 / ((F * K)**(1 - beta)) + 
                    (rho * beta * nu * alpha) / (4 * FK_beta) + 
                    ((2 - 3 * rho**2) / 24) * nu**2) * T

    return (numerator / denominator) * brackets

# ==========================================
# 3. HESTON MODEL (Fourier Transform)
# ==========================================
def heston_char_func(u, S0, K, T, r, v0, kappa, theta, sigma, rho):
    """
    Heston Characteristic Function.
    """
    xi = kappa - sigma * rho * 1j * u
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    
    g1 = (xi + d) / (xi - d)
    
    # Complex exponent terms
    exp_dt = np.exp(-d * T)
    
    term1 = np.exp(1j * u * (np.log(S0) + r * T))
    term2 = np.exp((kappa * theta) / sigma**2 * ((xi + d) * T - 2 * np.log((1 - g1 * exp_dt) / (1 - g1))))
    term3 = np.exp((v0 / sigma**2) * (xi + d) * (1 - exp_dt) / (1 - g1 * exp_dt))
    
    return term1 * term2 * term3

def heston_price_fourier(S0, K, T, r, v0, kappa, theta, sigma, rho):
    """
    Calculates Heston Call Price using numerical integration of characteristic functions (Gil-Pelaez formula).
    """
    # Integrand 1
    def integrand1(u):
        num = np.imag(np.exp(-1j * u * np.log(K)) * heston_char_func(u - 1j, S0, K, T, r, v0, kappa, theta, sigma, rho) / (S0 * np.exp(r * T)))
        return num / u

    # Integrand 2
    def integrand2(u):
        num = np.imag(np.exp(-1j * u * np.log(K)) * heston_char_func(u, S0, K, T, r, v0, kappa, theta, sigma, rho))
        return num / u

    # Integration limits (0 to infinity, practically 100 is enough for convergence usually)
    limit = 100 
    
    # Using scipy.integrate.quad
    P1 = 0.5 + (1 / np.pi) * quad(integrand1, 1e-8, limit)[0]
    P2 = 0.5 + (1 / np.pi) * quad(integrand2, 1e-8, limit)[0]
    
    CallPrice = S0 * P1 - K * np.exp(-r * T) * P2
    return max(CallPrice, 0.0)

# ==========================================
# 4. STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Quant Option Pricer", layout="wide")
st.title("⚡ Quant Option Pricing Engine")
st.markdown("Estimate option prices using **SABR** (Implied Volatility) and **Heston** (Stochastic Volatility).")

tabs = st.tabs(["SABR Model (Vol Surface)", "Heston Model (Pricing)"])

# --- TAB 1: SABR MODEL ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("SABR Parameters")
        sabr_F = st.number_input("Forward Rate (F)", value=100.0)
        sabr_K = st.number_input("Strike Price (K)", value=100.0)
        sabr_T = st.number_input("Time to Expiry (T)", value=1.0)
        
        st.markdown("---")
        sabr_alpha = st.slider("Alpha (Vol Level)", 0.01, 1.0, 0.2)
        sabr_beta = st.slider("Beta (Skew Backbone)", 0.0, 1.0, 0.5)
        sabr_rho = st.slider("Rho (Correlation)", -1.0, 1.0, -0.3)
        sabr_nu = st.slider("Nu (Vol of Vol)", 0.0, 2.0, 0.4)

    with col2:
        st.subheader("SABR Results")
        
        # Calculate Single Point
        iv = sabr_vol(sabr_F, sabr_K, sabr_T, sabr_alpha, sabr_beta, sabr_rho, sabr_nu)
        
        # Display Metrics
        m1, m2 = st.columns(2)
        m1.metric("Implied Volatility", f"{iv:.2%}")
        
        # If user wants a price, we assume r=0 for pure SABR usually, but let's allow a rough BS Calc
        price = bs_call_price(sabr_F, sabr_K, sabr_T, 0.0, iv) # Using F as S approx for simple display
        m2.metric("Est. Call Price (r=0)", f"${price:.2f}")

        # Plot Volatility Smile
        st.markdown("#### Volatility Smile")
        import matplotlib.pyplot as plt
        
        strikes = np.linspace(sabr_F * 0.5, sabr_F * 1.5, 50)
        vols = [sabr_vol(sabr_F, k, sabr_T, sabr_alpha, sabr_beta, sabr_rho, sabr_nu) for k in strikes]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(strikes, vols, label="SABR Smile", color="#FF4B4B", linewidth=2)
        ax.axvline(sabr_F, color="gray", linestyle="--", label="ATM (Forward)")
        ax.scatter([sabr_K], [iv], color="black", zorder=5, label="Current Strike")
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"SABR Volatility Smile (T={sabr_T})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# --- TAB 2: HESTON MODEL ---
with tabs[1]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Heston Parameters")
        h_S0 = st.number_input("Spot Price (S0)", value=100.0)
        h_K = st.number_input("Strike Price (K)", value=100.0, key="h_K")
        h_T = st.number_input("Time (T)", value=1.0, key="h_T")
        h_r = st.number_input("Risk-Free Rate (r)", value=0.03)
        
        st.markdown("---")
        h_v0 = st.number_input("Initial Variance (v0)", value=0.04)
        h_kappa = st.number_input("Mean Reversion (Kappa)", value=2.0)
        h_theta = st.number_input("Long-Run Variance (Theta)", value=0.04)
        h_sigma = st.number_input("Vol of Vol (Sigma)", value=0.3)
        h_rho = st.slider("Correlation (Rho)", -1.0, 1.0, -0.5, key="h_rho")

    with col2:
        st.subheader("Heston Pricing (Fourier Transform)")
        
        if st.button("Calculate Heston Price"):
            with st.spinner("Integrating Characteristic Function..."):
                h_price = heston_price_fourier(h_S0, h_K, h_T, h_r, h_v0, h_kappa, h_theta, h_sigma, h_rho)
                
                # Compare with BS
                bs_price = bs_call_price(h_S0, h_K, h_T, h_r, np.sqrt(h_v0))
                
                st.success(f"**Heston Call Price:** ${h_price:.4f}")
                
                st.markdown("### Comparison")
                comp_data = {
                    "Model": ["Heston", "Black-Scholes (Const Vol)"],
                    "Price": [h_price, bs_price],
                    "Assumption": ["Stochastic Volatility", f"Fixed Vol = {np.sqrt(h_v0):.2%}"]
                }
                st.table(comp_data)
                
                # Feller Condition Check
                feller = 2 * h_kappa * h_theta
                vol_sq = h_sigma ** 2
                st.info(f"**Feller Condition Check:** 2*kappa*theta ({feller:.4f}) > sigma^2 ({vol_sq:.4f})? {'✅ Yes' if feller > vol_sq else '⚠️ No (Simulation unstable)'}")
